import pandas as pd
from pathlib import Path
import numpy as np

judged_dir=Path('pairwise_outputs')
files=sorted(judged_dir.glob('judged_*.csv'))
if not files:
    print('No judged files found')
    raise SystemExit(0)

df=pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
# build pairs
rows=[]
for rk,g in df.groupby('round_key'):
    players=list(g['player_id'])
    scores=list(g['score'])
    valids=list(g['valid'])
    for i,pid in enumerate(players):
        for j,opp in enumerate(players):
            if i==j: continue
            rows.append({'round_key':rk,'player_id':pid,'opponent_id':opp,'score':int(scores[i]),'valid':int(valids[i])})

pairs=pd.DataFrame(rows)
# base mapping
pairs['base_player']=pairs['player_id'].map(lambda x: x.split('_self')[0])
pairs['base_opponent']=pairs['opponent_id'].map(lambda x: x.split('_self')[0])
self_pairs=pairs[(pairs['base_player']==pairs['base_opponent']) & (pairs['player_id']!=pairs['opponent_id'])]
if self_pairs.empty:
    print('No self pairs')
    raise SystemExit(0)

out=[]
for base in sorted(self_pairs['base_player'].unique()):
    sub=self_pairs[self_pairs['base_player']==base]
    rounds=sorted(sub['round_key'].unique())
    a_means=[]; b_means=[]; both_valids=[]; both_zero=[]; count=0
    for rk in rounds:
        gr=sub[sub['round_key']==rk]
        try:
            a_row=gr[gr['player_id'].str.endswith('_selfA')].iloc[0]
            b_row=gr[gr['player_id'].str.endswith('_selfB')].iloc[0]
        except Exception:
            continue
        a_means.append(a_row['score'])
        b_means.append(b_row['score'])
        both_valids.append(1 if (a_row['valid']==1 and b_row['valid']==1) else 0)
        both_zero.append(1 if (a_row['score']==0 and b_row['score']==0) else 0)
        count+=1
    if count==0:
        continue
    out.append({'base':base,'rounds':count,'mean_A_vs_B':np.mean(a_means),'mean_B_vs_A':np.mean(b_means),'sum_means':np.mean(a_means)+np.mean(b_means),'both_valid_frac':np.mean(both_valids),'both_zero_frac':np.mean(both_zero)})

print('base,rounds,mean_A_vs_B,mean_B_vs_A,sum_means,both_valid_frac,both_zero_frac')
for r in out:
    print(f"{r['base']},{r['rounds']},{r['mean_A_vs_B']:.6f},{r['mean_B_vs_A']:.6f},{r['sum_means']:.6f},{r['both_valid_frac']:.6f},{r['both_zero_frac']:.6f}")
