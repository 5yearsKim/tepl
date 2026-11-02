#!/usr/bin/env python3
"""Export the pinned public Flax model and inventory unmodified XLA HLOs."""
from pathlib import Path
import hashlib, json, sys, os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import jax, jax.numpy as jnp
import flax_lm1b_models as models
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
config=dict(vocab_size=128,output_vocab_size=128,emb_dim=32,qkv_dim=32,
            mlp_dim=64,num_heads=2,num_layers=1,max_len=16,
            dropout_rate=0.,attention_dropout_rate=0.,deterministic=True,decode=False)
m=models.TransformerLM(models.TransformerConfig(**config))
rows=[]
for batch in [1,4]:
 tokens=(jnp.arange(batch*16,dtype=jnp.int32).reshape(batch,16)%127)+1
 params=m.init(jax.random.key(0),tokens)
 f=jax.jit(lambda p,t:m.apply(p,t))
 hlo=f.lower(params,tokens).compiler_ir('hlo').as_hlo_text()
 name=f'transformer_b{batch}_l16';p=out/(name+'.hlo');p.write_text(hlo)
 rows.append(dict(name=name,model='Flax TransformerLM',batch=batch,sequence_length=16,
                  input=str(p),config=config,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
xla=Path(os.environ.get('XLA_ROOT','/home/onion/Projects/xla'))/'xla/backends/cpu/benchmarks/hlo'
for name,filename in [('dlrm','dlrm_keras_jax_f32_train_step.hlo'),('resnet50','resnet50_keras_jax_f32.hlo')]:
 p=out/(name+'.hlo');p.write_bytes((xla/filename).read_bytes())
 rows.append(dict(name=name,model=name,batch='original',input=str(p),
                  upstream=str(xla/filename),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(out/'workloads.json').write_text(json.dumps(dict(jax=jax.__version__,config=config,workloads=rows),indent=2)+'\n')
print(json.dumps([dict(name=r['name'],bytes=Path(r['input']).stat().st_size) for r in rows],indent=2))
