from pathlib import Path
import pandas as pd
from quinte_v2 import load_artifact
from quinte_v3 import load_ranker
from quinte_v43 import load_outsider_artifact
from quinte_v44 import rank_quinte_v44
from quinte_v45 import apply_v45_market_dynamics
from quinte_v46 import load_fundamental_artifact, add_v46_fundamental
from quinte_v51 import load_v51_artifact, add_v51_meta_consensus
BASE=Path(__file__).resolve().parent
h=pd.read_csv(BASE/'validated_history.csv')
h['race_date']=pd.to_datetime(h['race_date'])
target=pd.Timestamp('2026-09-06')
rid='R1C3_2026-09-06'
race=h[(h.race_id==rid)].copy()
hist=h[h.race_date<target].copy()
a=load_artifact(BASE/'quinte_v2_artifact.json')
r=load_ranker(BASE/'quinte_v2_ranker.txt')
o=load_outsider_artifact(BASE/'v43_outsider_artifact.json')
fa=load_fundamental_artifact(BASE/'v46_fundamental_artifact.json')
fr=load_ranker(BASE/'v46_fundamental_ranker.txt')
ma=load_v51_artifact(BASE/'v51_meta_artifact.json')
v44=rank_quinte_v44(hist,race,a,r,o)
v45=apply_v45_market_dynamics(v44)
v46=add_v46_fundamental(hist,race,v45,fa,fr)
v51=add_v51_meta_consensus(v46,ma)
assert v51[v51.selected_top7].horse_number.astype(int).tolist()==v46[v46.selected_top7].horse_number.astype(int).tolist()
assert 'v51_meta_top5_score' in v51
assert v51.v51_consensus_confidence.notna().all()
print('OFFICIAL',v51[v51.selected_top7].horse_number.astype(int).tolist())
print('FUND',v51.sort_values('v46_fundamental_rank').head(7).horse_number.astype(int).tolist())
print('SHADOW',v51[v51.v51_shadow_top7].sort_values('v51_shadow_rank').horse_number.astype(int).tolist())
print('TOP3',v51[v51.v51_shadow_top3].sort_values('v51_shadow_rank').horse_number.astype(int).tolist())
print('CONF',float(v51.iloc[0].v51_consensus_confidence),'DIV',float(v51.iloc[0].v51_divergence_index),'STANCE',v51.iloc[0].v51_stance)
print('SWAP IN',v51.iloc[0].v51_shadow_swap_in,'OUT',v51.iloc[0].v51_shadow_swap_out)
from etrio_v51 import generate_etrio_10
etrio, pool = generate_etrio_10(v51)
assert len(etrio) == 10, len(etrio)
top5 = set(v51.sort_values('quinte_rank').head(5).horse_number.astype(int).tolist())
assert len(set(etrio['Combinaison'])) == 10
for _, rr in etrio.iterrows():
    trio = {int(rr['Base 1']), int(rr['Base 2']), int(rr['Associé'])}
    assert len(trio & top5) == 2, (trio, top5)
    assert int(rr['Associé']) not in top5
print('ETRIO_BASES', sorted(top5))
print('ETRIO_ASSOC', pool.horse_number.astype(int).tolist())
print(etrio.to_string(index=False))
from quinte70_v51 import generate_quinte_70
q70, exp70, scen70 = generate_quinte_70(v51)
assert len(q70) == 70, len(q70)
assert q70['Combinaison'].nunique() == 70
pool10 = set(exp70['N°'].astype(int).tolist())
assert len(pool10) == 10
for _, rr in q70.iterrows():
    vals = {int(rr[c]) for c in ['C1','C2','C3','C4','C5']}
    assert len(vals) == 5
    assert vals <= pool10
assert 'Indice historique' in q70.columns
assert 'Score combiné' in q70.columns
print('QUINTE70_RETRO_POOL', exp70.sort_values('Rang pool')['N°'].astype(int).tolist())
print(scen70.to_string(index=False))
print(exp70[['Rang pool','N°','Présence /70','Force intelligente']].to_string(index=False))
from adaptive_tickets_v52 import select_adaptive_tickets
qadapt, dec = select_adaptive_tickets(q70, v51)
assert dec['ticket_count'] in (30,50,70)
assert len(qadapt) == dec['ticket_count']
assert qadapt['Combinaison'].nunique() == len(qadapt)
print('V52_PROFILE', dec['profile'], 'TICKETS', dec['ticket_count'], 'STAKE', dec['full_stake_eur'])
