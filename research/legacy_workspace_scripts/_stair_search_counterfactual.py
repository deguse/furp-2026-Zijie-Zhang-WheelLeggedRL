import hashlib, json, os
from hoppertrex_mjlab.scripts.rsl_rl.stair_dynamic_search_live_adapter import _score_batch
p=os.environ['HOPPERTREX_DYNAMIC_STAIR_STAGE5_CHECKPOINT_PATH']
h=hashlib.sha256(open(p,'rb').read()).hexdigest()
candidates=[
 [0.0,0.02,0.0,2.0],
 [0.0,0.07,0.0,2.0],
 [0.035,0.045,0.20,2.0],
 [0.07,0.07,0.40,2.0],
 [0.035,0.07,0.0,2.0],
 [0.0,0.045,0.20,2.0],
]
out={}
for family in ('synchronized','alternating'):
 out[family]=_score_batch(family=family,candidates=candidates,replicates=8,device='cpu',expected_stage5_checkpoint_sha256=h)
print('COUNTERFACTUAL_JSON_START')
print(json.dumps({'candidates':candidates,'scores':out},indent=2,sort_keys=True))
