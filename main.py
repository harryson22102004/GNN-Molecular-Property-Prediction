import torch, torch.nn as nn, torch.nn.functional as F
class MPNNLayer(nn.Module):
    def __init__(self, nd, ed, od):
        super().__init__()
        self.msg=nn.Sequential(nn.Linear(2*nd+ed,od),nn.ReLU(),nn.Linear(od,od))
        self.upd=nn.GRUCell(od,nd)
    def forward(self,x,ei,ea):
        s,d=ei
        m=self.msg(torch.cat([x[s],x[d],ea],-1))
        agg=torch.zeros(x.size(0),m.size(-1),device=x.device)
        agg.index_add_(0,d,m)
        return self.upd(agg,x)
 
class MolGNN(nn.Module):
    def __init__(self, nd=64, ed=16, nl=3, out=1):
        super().__init__()
        self.ne=nn.Linear(9,nd); self.ee=nn.Linear(4,ed)
        self.layers=nn.ModuleList([MPNNLayer(nd,ed,nd) for _ in range(nl)])
        self.readout=nn.Sequential(nn.Linear(nd,128),nn.ReLU(),nn.Linear(128,out))
    def forward(self,x,ei,ea,batch):
        x=self.ne(x.float()); e=self.ee(ea.float())
        for l in self.layers: x=F.relu(l(x,ei,e))
        pool=torch.zeros(batch.max()+1,x.size(-1),device=x.device)
        pool.index_add_(0,batch,x)
        cnt=torch.bincount(batch).float().unsqueeze(1)
        return self.readout(pool/cnt)
 
model=MolGNN()
x=torch.randint(0,5,(20,9)); ei=torch.randint(0,20,(2,40))
ea=torch.randint(0,3,(40,4)); batch=torch.zeros(20,dtype=torch.long)
out=model(x,ei,ea,batch)
print(f"Molecular property prediction: {out.item():.3f}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
