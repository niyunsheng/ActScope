import type { Model } from './engine';
const tensor = (name:string, shape:string, kind:'replicated'|'tp'|'expert', axis=0,shardUnit='1') => ({id:name,name,shape,kind,axis,shardUnit,trainable:true});
export function preset(kind: string): Model {
  const moe = kind === 'moe';
  if (kind === 'custom') return {version:1,name:'Custom module',variables:[{name:'hidden_size',value:1024},{name:'intermediate_size',value:4096},{name:'num_layers',value:1},{name:'tokens_per_rank',value:1024}],repeats:'num_layers',tensors:[tensor('linear.weight','intermediate_size, hidden_size','tp')],activations:[]};
  return { version:1, name:moe?'Qwen3-235B-A22B':'Qwen3-8B', repeats:'num_layers',
    variables: Object.entries({hidden_size:4096,num_layers:moe?94:36,num_attention_heads:moe?64:32,num_key_value_heads:moe?4:8,head_dim:128,...(moe?{moe_intermediate_size:1536,num_experts:128,top_k:8}:{intermediate_size:12288}),tokens_per_rank:1024}).map(([name,value])=>({name,value})),
    tensors:[tensor('input_layernorm.weight','hidden_size','replicated'),tensor('post_attention_layernorm.weight','hidden_size','replicated'),tensor('attention.q_norm.weight','head_dim','replicated'),tensor('attention.k_norm.weight','head_dim','replicated'),tensor('attention.q_proj.weight','num_attention_heads * head_dim, hidden_size','tp',0,'head_dim'),tensor('attention.k_proj.weight','num_key_value_heads * head_dim, hidden_size','tp',0,'head_dim'),tensor('attention.v_proj.weight','num_key_value_heads * head_dim, hidden_size','tp',0,'head_dim'),tensor('attention.o_proj.weight','hidden_size, num_attention_heads * head_dim','tp',1,'head_dim'),
    ...(moe?[tensor('moe.router.weight','num_experts, hidden_size','replicated'),tensor('experts.gate_proj.weight','num_experts, moe_intermediate_size, hidden_size','expert',1),tensor('experts.up_proj.weight','num_experts, moe_intermediate_size, hidden_size','expert',1),tensor('experts.down_proj.weight','num_experts, hidden_size, moe_intermediate_size','expert',2)]:[tensor('mlp.gate_proj.weight','intermediate_size, hidden_size','tp'),tensor('mlp.up_proj.weight','intermediate_size, hidden_size','tp'),tensor('mlp.down_proj.weight','hidden_size, intermediate_size','tp',1)])],
    // Intentionally an empty ledger: no unverified whole-layer activation coefficients.
    activations:[] };
}
export const sources = {moe:'https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json',dense:'https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json'};
