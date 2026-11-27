"""Provisional Majorah preference score, not a return forecast or fitted model.
Canonical inputs: percentages in percentage points, ratios in multiples.
Run: python majorah_score_v1.py input.csv output.json
"""
import csv, json, math, sys
from statistics import median

VERSION = '1.0-heuristic'
WEIGHTS = dict(health=.25, valuation=.25, trajectory=.20, fear=.15, profitability=.10, shareholder=.05)

def number(x):
    try:
        x=float(x)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError): return None

def curve(x, knots):
    x=number(x)
    if x is None: return None
    if x <= knots[0][0]: return knots[0][1]
    for (a,b),(c,d) in zip(knots,knots[1:]):
        if x <= c: return b+(d-b)*(x-a)/(c-a)
    return knots[-1][1]

def history(x):
    try:
        a=json.loads(x) if isinstance(x,str) else x
        return [number(v) for v in a] if isinstance(a,list) else []
    except (ValueError,TypeError): return []

def score(row):
    def n(k): return number(row.get(k))
    def c(k,knots): return curve(n(k),knots)
    def positive_ratio(k,knots):
        v=n(k); return curve(v,knots) if v is not None and v>0 else None
    def delta(k):
        a=history(row.get(k)); return a[-1]-a[-3] if len(a)>=3 and a[-1] is not None and a[-3] is not None else None
    def hist_score(k,h):
        v=n(k); a=[x for x in history(row.get(h)) if x is not None and x>0]
        return curve(v/median(a),[(.5,100),(.8,85),(1,55),(1.3,25),(2,0)]) if v is not None and v>0 and len(a)>=4 else None
    growth=[(-40,0),(-15,20),(-5,40),(0,55),(5,75),(15,92),(40,100)]
    groups={
      'health':[(c('net_debt_ebitda',[(-2,100),(0,90),(1,76),(2,60),(3,38),(5,10),(7,0)]),.35),(c('net_cash_market_cap_pct',[(-60,0),(-30,25),(-10,60),(0,80),(20,95),(40,100)]),.25),(c('current_ratio',[(.5,0),(1,40),(1.5,75),(2.5,100)]),.20),(c('interest_coverage',[(0,0),(1,10),(2,35),(5,70),(15,95),(30,100)]),.20)],
      'valuation':[(positive_ratio('forward_pe',[(5,100),(10,92),(15,75),(20,52),(30,15),(45,0)]),.22),(positive_ratio('ptbv',[(.5,100),(1,95),(2,75),(3,55),(5,30),(9,5)]),.23),(positive_ratio('forward_ps',[(.3,100),(1,85),(3,55),(6,20),(10,0)]),.10),(hist_score('forward_pe','forward_pe_history_oldest_first'),.20),(hist_score('ptbv','ptbv_history_oldest_first'),.15),(hist_score('pb','pb_history_oldest_first'),.10)],
      'trajectory':[(c('revenue_growth_yoy_pct',growth),.30),(c('earnings_growth_yoy_pct',growth),.20),(c('fcf_growth_yoy_pct',growth),.15),(c('net_cash_assets_change_pp',[(-10,0),(-3,30),(0,55),(3,85),(8,100)]),.15),(curve(delta('operating_margin_history_oldest_first'),[(-8,0),(-3,25),(0,55),(3,85),(6,100)]),.20)],
      'fear':[(c('rsi',[(15,100),(25,95),(35,75),(50,40),(70,0)]),.30),(c('above_52w_low_pct',[(0,100),(10,90),(25,55),(60,0)]),.30),(c('above_3y_low_pct',[(0,100),(10,95),(25,70),(60,25),(100,0)]),.40)],
      'profitability':[(c('fcf_margin_pct',[(-20,0),(0,30),(5,60),(10,78),(20,95),(30,100)]),.60),(curve((history(row.get('operating_margin_history_oldest_first')) or [None])[-1],[(0,0),(5,40),(10,65),(20,90),(35,100)]),.40)],
      'shareholder':[(c('net_share_change_1y_pct',[(-8,100),(-3,95),(0,70),(3,40),(6,20),(15,0)]),1)]}
    dims={}; cov={}
    for k,parts in groups.items():
        cov[k]=sum(w for v,w in parts if v is not None)
        dims[k]=sum(v*w for v,w in parts if v is not None)/cov[k] if cov[k] else None
    coverage=sum(WEIGHTS[k]*cov[k] for k in WEIGHTS)
    # Missing data are not assumed favorable. Require health, valuation, trajectory and fear.
    if coverage<.75 or any(cov[k]<.5 for k in ('health','valuation','trajectory','fear')):
        return dict(version=VERSION,score=None,coverage=round(coverage,3),dimensions=dims,adjustments={},status='insufficient_data')
    # Neutral shrinkage for incomplete dimensions.
    base=sum(WEIGHTS[k]*((dims[k] or 0)*cov[k]+50*(1-cov[k])) for k in WEIGHTS)
    def ramp(v,a,b): return 0 if v is None else max(0,min(1,(v-a)/(b-a)))
    h,v,t,f,p,s=[dims[k] for k in ('health','valuation','trajectory','fear','profitability','shareholder')]
    adj={
      'fundamental_valuation':10*ramp(t,55,90)*ramp(v,55,95),
      'net_cash_buybacks_growth':5*ramp(n('net_cash_market_cap_pct'),0,20)*ramp(n('net_share_change_1y_pct'),0,-4)*ramp(n('revenue_growth_yoy_pct'),0,10),
      'quality_valuation':5*ramp(p,65,95)*ramp(v,60,95),
      'beta_fear_quality':5*ramp(n('beta'),1,2.5)*ramp(f,75,100)*ramp(h,75,95)*ramp(t,70,95)*ramp(v,70,95)*ramp(n('market_drawdown_pct'),10,30),
      'leverage_margin_stress':-15*ramp(n('net_debt_ebitda'),1.5,5)*ramp(delta('operating_margin_history_oldest_first'),0,-5),
      'cash_liquidity_stress':-15*ramp(n('fcf_margin_pct'),0,-10)*ramp(n('current_ratio'),1.5,.7),
      'dilution_health_stress':-10*ramp(n('net_share_change_1y_pct'),0,12)*ramp(h,65,20)}
    return dict(version=VERSION,score=round(max(0,min(100,base+sum(adj.values()))),2),coverage=round(coverage,3),dimensions={k:None if x is None else round(x,2) for k,x in dims.items()},base=round(base,2),adjustments={k:round(x,2) for k,x in adj.items()},status='provisional')

if __name__=='__main__':
    if len(sys.argv)!=3: raise SystemExit(__doc__)
    with open(sys.argv[1],newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    results=[]
    for row in rows:
        r=score(row); results.append(dict(case_id=row.get('case_id') or row.get('ticker'),actual=row.get('score'),**r))
    with open(sys.argv[2],'w',encoding='utf-8') as f: json.dump(results,f,indent=2,allow_nan=False)
    errors=[abs(r['score']-float(r['actual'])) for r in results if r['score'] is not None and number(r['actual']) is not None]
    print(f'Scored {sum(r["score"] is not None for r in results)}/{len(results)} rows; retrospective MAE: {sum(errors)/len(errors):.2f}' if errors else f'Processed {len(results)} rows')
