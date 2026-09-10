"""
scent_app.py —— ScentAI 后端 (FastAPI)
运行:
  pip install -r requirements.txt
  uvicorn scent_app:app --host 0.0.0.0 --port $PORT
接口:
  GET  /               健康检查
  GET  /catalog        品牌->香水名 (前端下拉用)
  POST /recommend_split  双栏推荐 + NLP 场景描述
可选: 设环境变量 HF_TOKEN 后, 描述改用 Hugging Face 大模型生成 (否则用模板)
"""
import os, re, random
import numpy as np, pandas as pd
from collections import Counter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

DATA_DIR="."
BIG5=["openness","conscientiousness","extraversion","agreeableness","neuroticism"]
ACC=["accord1","accord2","accord3","accord4","accord5"]
POS=np.array([1.0,0.8,0.6,0.4,0.3]); ALL_TIERS=["commercial","premium","niche","ultra niche"]
def _norm(s): return re.sub(r"[^a-z0-9]+"," ",str(s).lower()).strip() if pd.notna(s) else ""

# ================= NLP 场景描述 =================
MBTI_DESC={"INTJ":"an architect's far-seeing mind","INTP":"a restless, questioning intellect",
"ENTJ":"a commander's forward drive","ENTP":"a spark-chasing curiosity","INFJ":"a quiet, knowing depth",
"INFP":"a dreamer's tender interior","ENFJ":"a warm, magnetic pull","ENFP":"an open, electric enthusiasm",
"ISTJ":"a grounded sense of order","ISFJ":"a soft, steady devotion","ESTJ":"a decisive, capable presence",
"ESFJ":"a generous, people-warmed heart","ISTP":"a cool, hands-on precision","ISFP":"a private, artful sensitivity",
"ESTP":"a bold appetite for the moment","ESFP":"a bright, in-the-room radiance"}
ZODIAC_DESC={"Aries":"fire and forward motion","Taurus":"earthbound, sensual patience",
"Gemini":"quicksilver curiosity","Cancer":"tidal, protective feeling","Leo":"sunlit, generous pride",
"Virgo":"precise, discerning care","Libra":"poised, harmony-seeking grace","Scorpio":"deep, magnetic intensity",
"Sagittarius":"restless, horizon-chasing fire","Capricorn":"cool, ascending resolve",
"Aquarius":"an unconventional current","Pisces":"a dreaming, boundless tide"}
MOOD_LINE={"confident":"Today you want presence — a scent that speaks a beat before you do.",
"calm":"Today you want stillness — something that lowers the volume of the room.",
"uplifted":"Today you want lift — a brightness that follows you like light.",
"comforted":"Today you want warmth — the olfactory equivalent of coming home."}
SCENE_OPEN={"date":"For an evening you want remembered","work":"For the hours that ask you to be sharp",
"home":"For the slow, unhurried hours at home","travel":"For a day spent in motion",
"dining":"For a table shared with people who matter"}
ACCORD_IMG={"woody":"warm cedar and dry sandalwood","oud":"smoky oud and resin","amber":"golden amber",
"vanilla":"soft vanilla","musky":"clean, skin-like musk","musk":"clean, skin-like musk","leather":"supple leather",
"leathery":"supple leather","citrus":"bright citrus","floral":"blooming florals","white floral":"heady white flowers",
"rose":"velvety rose","fruity":"ripe, sunlit fruit","fresh spicy":"cool, snapping spice","warm spicy":"glowing spice",
"soft spicy":"soft spice","spicy":"spice","aquatic":"clean sea air","green":"crushed green leaves",
"powdery":"soft powder","smoky":"curling smoke","iris":"cool iris","patchouli":"earthy patchouli",
"tobacco":"sweet tobacco","sweet":"gourmand sweetness","aromatic":"herbal aromatics","earthy":"damp earth",
"balsamic":"warm balsam","ozonic":"clean ozone","fresh":"airy freshness","tropical":"lush tropical fruit",
"honey":"warm honey","caramel":"burnt caramel","almond":"soft almond","coconut":"creamy coconut",
"tea":"steeped tea","aldehydic":"soapy aldehydes","animalic":"warm, animalic depth"}
ATMOS=["it settles close to the skin and stays.","warm, deliberate, and a little unexpected.",
"familiar enough to trust, strange enough to remember.","the kind of trail people follow without knowing why.",
"quiet at first, then impossible to ignore."]
CLOSERS=["This is a scent that doesn't ask permission.","Wear it, and the room rearranges itself around you.",
"It reads as intention, never accident.","Let it be the thing people can't quite place about you.",
"You won't so much wear it as become it."]

def _describe_template(mbti,zodiac,scene,mood,accords):
    imgs=[ACCORD_IMG.get(a,a) for a in accords[:3]] or ["something quietly distinctive"]
    if len(imgs)==1: img_list=imgs[0]
    elif len(imgs)==2: img_list=f"{imgs[0]} and {imgs[1]}"
    else: img_list=f"{imgs[0]}, {imgs[1]} and {imgs[2]}"
    opener=SCENE_OPEN.get(scene,"For the day ahead")
    zd=ZODIAC_DESC.get(zodiac,"your own rhythm"); md=MBTI_DESC.get(mbti,"a mind all your own")
    ml=MOOD_LINE.get(mood,"Today you want something that simply feels like you.")
    return (f"{opener}, your {zd} meets {md}. {ml} "
            f"The reading gathers around {img_list} — {random.choice(ATMOS)} {random.choice(CLOSERS)}")

def _describe_hf(token,mbti,zodiac,scene,mood,accords):
    from huggingface_hub import InferenceClient
    client=InferenceClient(token=token)
    prompt=("Write an evocative, second-person fragrance description of at most 200 words. "
            f"The person's MBTI is {mbti}, star sign {zodiac}, dressing for {scene}, feeling {mood}. "
            f"The recommended perfume features notes of {', '.join(accords)}. "
            "Poetic but grounded, no bullet lists, no headings.")
    out=client.chat_completion(messages=[{"role":"user","content":prompt}],
        model="mistralai/Mistral-7B-Instruct-v0.3",max_tokens=300,temperature=0.8)
    return out.choices[0].message.content.strip()

def describe(mbti,zodiac,scene,mood,accords):
    token=os.environ.get("HF_TOKEN")
    if token:
        try: return _describe_hf(token,mbti,zodiac,scene,mood,accords)
        except Exception as e: print("HF generation failed, using template:",e)
    return _describe_template(mbti,zodiac,scene,mood,accords)

# ================= 推荐引擎 =================
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
        # 品牌->香水名 目录 (前端下拉)
        self.catalog={}
        for b,n in zip(self.prod["Brand"].astype(str),self.prod["Name"].astype(str)):
            self.catalog.setdefault(b,[]).append(n)
        for b in self.catalog: self.catalog[b]=sorted(set(self.catalog[b]))
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
        new_out=[]; top=[]
        if allow.sum()>0:
            b=np.where(allow,score_new,-1e9); ok=b>-1e8
            bn=np.where(ok,(b-b[ok].min())/(b[ok].max()-b[ok].min()+1e-9),0)
            final=np.where(ok,(1-rating_w)*bn+rating_w*rn,-1e9); top=np.argsort(-final)[:topn]
            for r in top: new_out.append(row(int(r),"match",final[r]))
        # 汇总top推荐的主香调 -> 生成描述
        cnt=Counter()
        for r in list(top):
            for c in ACC:
                v=self.prod.iloc[int(r)][c]
                if pd.notna(v): cnt[str(v)]+=1
        top_accords=[a for a,_ in cnt.most_common(4)] or ["something quietly distinctive"]
        desc=describe(mt,zs,scene,feel,top_accords)
        return {"owned":owned_out,"new":new_out,"description":desc,"accords":top_accords}

app=FastAPI(title="ScentAI")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
rec=ScentRecommender(DATA_DIR)

class Req(BaseModel):
    mbti:str; zodiac:str; scene:str; mood:str
    gender:str="any"; tiers:Optional[List[str]]=None
    purchased:List[str]=[]; purchase_w:float=0.5; topn:int=6

@app.get("/")
def health(): return {"status":"ScentAI API is running","products":int(len(rec.prod))}

@app.get("/catalog")
def catalog(): return rec.catalog

@app.post("/recommend_split")
def recommend_split(r:Req):
    purchased=rec.match_purchases(r.purchased)
    return rec.recommend_split(r.mbti,r.zodiac,r.scene,r.mood,purchased,
        gender_pref=r.gender,tiers=r.tiers,purchase_w=r.purchase_w,topn=r.topn)        M=np.zeros((len(self.prod),len(self.A)),np.float32)
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
