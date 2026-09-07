"""
scent_app.py —— ScentAI 后端 (FastAPI)
把 notebook 里的推荐引擎包成 API 供前端调用。
运行:
  pip install fastapi uvicorn pandas numpy openpyxl
  mkdir static && cp scent_frontend.html static/index.html
  uvicorn scent_app:app --reload --port 8000
  # 打开 http://localhost:8000
数据(同目录): mbti_big5_map.csv, zodiac_big5_map.csv,
  accord_semantic_map_clean.xlsx, accord_scene_map_clean.xlsx,
  accord_mood_map_clean.xlsx, perfumes_tagged.xlsx
"""
import re, numpy as np, pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

DATA_DIR="."
BIG5=["openness","conscientiousness","extraversion","agreeableness","neuroticism"]
ACC=["accord1","accord2","accord3","accord4","accord5"]
POS=np.array([1.0,0.8,0.6,0.4,0.3]); ALL_TIERS=["commercial","premium","niche","ultra niche"]
def _norm(s): return re.sub(r"[^a-z0-9]+"," ",str(s).lower()).strip() if pd.notna(s) else ""

class ScentRecommender:
    def __init__(self,d="."):
        self.mbti=pd.read_csv(f"{d}/mbti_big5_map.csv"); self.zod=pd.read_csv(f"{d}/zodiac_big5_map.csv")
        sem=pd.read_excel(f"{d}/accord_semantic_map_clean.xlsx")
        scn=pd.read_excel(f"{d}/accord_scene_map_clean.xlsx")
        mood=pd.read_excel(f"{d}/accord_mood_map_clean.xlsx")
        self.prod=pd.read_excel(f"{d}/perfumes_tagged.xlsx").reset_index(drop=True)
        self.A=sem["accord"].str.strip().tolist(); idx={a:i for i,a in enumerate(self.A)}
        self.S=sem[BIG5].values; self.C=sem["confidence"].values
        self.scn=scn.set_index(scn["accord"].str.strip()).loc[self.A,["work","home","travel","date","dining"]]
        self.mood=mood.set_index(mood["accord"].str.strip()).loc[self.A,["calm","confident","uplifted","comforted"]]
        self.rating=self.prod["rating"].values; self.gender=self.prod["gender"].values; self.tier=self.prod["tier"].values
        M=np.zeros((len(self.prod),len(self.A)),np.float32)
        for j,c in enumerate(ACC):
            codes=self.prod[c].map(lambda a: idx.get(str(a).strip(),-1) if pd.notna(a) else -1).values
            for r,cd in enumerate(codes):
                if cd>=0: M[r,cd]+=POS[j]
        self.M=M; self.Mn=M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
        self.pkey=(self.prod["Brand"].map(_norm)+" "+self.prod["Name"].map(_norm)).values
    def _pers_w(self,mt,zs,scene,feel,w_mbti,soft):
        u=(w_mbti*self.mbti.loc[self.mbti.mbti_type==mt,BIG5].values[0]
           +(1-w_mbti)*self.zod.loc[self.zod.zodiac_sign==zs,BIG5].values[0])-0.5
        wp=(self.S@u)*self.C; wp=wp/(np.abs(wp).max()+1e-9)
        return wp, np.where(self.scn[scene].values>0,1.0,soft), self.mood[feel].values
    def _accords_str(self,r):
        return " \u00B7 ".join(str(self.prod.iloc[r][c]) for c in ACC if pd.notna(self.prod.iloc[r][c]))
    def match_purchases(self,queries):
        got=[]
        for q in queries:
            toks=_norm(q).split()
            if not toks: continue
            hits=[i for i,s in enumerate(self.pkey) if all(t in s for t in toks)]
            if hits: got.append(min(hits,key=lambda i:len(str(self.prod.iloc[i]["Name"]))))
        return sorted(set(got))
    def recommend_split(self,mt,zs,scene,feel,purchased,gender_pref="any",tiers=None,
                        purchase_w=0.5,topn=8,w_mbti=0.7,soft=0.3,rating_w=0.15):
        wp,sf,mv=self._pers_w(mt,zs,scene,feel,w_mbti,soft); owned=set(purchased)
        wt=(self.M[list(owned)].sum(0)/(self.M[list(owned)].sum(0).max()+1e-9)) if owned else np.zeros(len(self.A))
        score_occ=self.Mn@(wp*sf*mv)
        w_new=(((1-purchase_w)*wp+purchase_w*wt)*sf*mv) if owned else wp*sf*mv
        score_new=self.Mn@w_new; rn=np.clip((self.rating-3.5)/1.5,0,1)
        def row(r,metric,val):
            return {"Name":self.prod.iloc[r]["Name"],"Brand":self.prod.iloc[r]["Brand"],
                    "tier":self.prod.iloc[r]["tier"],"gender":self.prod.iloc[r]["gender"],
                    "rating":round(float(self.prod.iloc[r]["rating"]),2),
                    "accords":self._accords_str(r),metric:round(float(val),3)}
        owned_out=[]
        if owned:
            oi=np.array(sorted(owned)); order=oi[np.argsort(-score_occ[oi])]
            owned_out=[row(int(r),"fit",score_occ[r]) for r in order]
        allow=np.ones(len(self.prod),bool)
        if owned: allow[list(owned)]=False
        if gender_pref!="any": allow&=(self.gender==gender_pref)|(self.gender=="unisex")
        if tiers and set(tiers)!=set(ALL_TIERS): allow&=np.isin(self.tier,list(tiers))
        new_out=[]
        if allow.sum()>0:
            b=np.where(allow,score_new,-1e9); ok=b>-1e8
            bn=np.where(ok,(b-b[ok].min())/(b[ok].max()-b[ok].min()+1e-9),0)
            final=np.where(ok,(1-rating_w)*bn+rating_w*rn,-1e9)
            for r in np.argsort(-final)[:topn]: new_out.append(row(int(r),"match",final[r]))
        return {"owned":owned_out,"new":new_out}

app=FastAPI(title="ScentAI")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
rec=ScentRecommender(DATA_DIR)

class Req(BaseModel):
    mbti:str; zodiac:str; scene:str; mood:str
    gender:str="any"; tiers:Optional[List[str]]=None
    purchased:List[str]=[]; purchase_w:float=0.5; topn:int=6

@app.post("/recommend_split")
def recommend_split(r:Req):
    purchased=rec.match_purchases(r.purchased)
    return rec.recommend_split(r.mbti,r.zodiac,r.scene,r.mood,purchased,
        gender_pref=r.gender,tiers=r.tiers,purchase_w=r.purchase_w,topn=r.topn)

@app.get("/")
def health():
    return {"status":"ScentAI API is running","products":int(len(rec.prod))}
