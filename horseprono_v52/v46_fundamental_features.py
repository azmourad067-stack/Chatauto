import numpy as np
import pandas as pd
from quinte_candidate import prepare, music_score, key

def _dist_bucket(s):
    d=pd.to_numeric(s,errors='coerce').fillna(2000)
    return pd.cut(d, bins=[-np.inf,1400,1800,2200,2600,np.inf], labels=['SPRINT','MILE','MID','STAYER','LONG']).astype(str)

def _rankpct(v, group, ascending=False):
    s=pd.to_numeric(v,errors='coerce')
    r=s.groupby(group).rank(method='average',ascending=ascending)
    n=s.groupby(group).transform('size').astype(float)
    return pd.Series(np.where(n>1,1-(r-1)/(n-1),1.0),index=s.index)

def _cum_prior(h, qq, entity_col, context_col, prefix, alpha_top=5.0, beta_top=5.0, alpha_win=1.0, beta_win=9.0):
    hh=h.copy()
    hh['race_date']=pd.to_datetime(hh.race_date).dt.normalize()
    qqd=pd.to_datetime(qq.race_date).dt.normalize()
    ent=key(hh[entity_col])
    qent=key(qq[entity_col])
    ctx=hh[context_col].fillna('').astype(str).str.strip().str.upper()
    qctx=qq[context_col].fillna('').astype(str).str.strip().str.upper()
    hh['_k']=ent+'||'+ctx
    qk=qent+'||'+qctx
    hh['_top5']=pd.to_numeric(hh.finish_position,errors='coerce').between(1,5).astype(float)
    hh['_win']=pd.to_numeric(hh.finish_position,errors='coerce').eq(1).astype(float)
    hh['_start']=1.0
    daily=(hh[hh['_k'].str.len()>2].groupby(['_k','race_date'])[['_start','_top5','_win']].sum().sort_index())
    if daily.empty:
        return pd.DataFrame({prefix+'_starts':np.zeros(len(qq)),prefix+'_top5':np.full(len(qq),alpha_top/(alpha_top+beta_top)),prefix+'_win':np.full(len(qq),alpha_win/(alpha_win+beta_win))})
    daily=daily.groupby(level=0).cumsum().reset_index()
    left=pd.DataFrame({'_k':qk,'race_date':qqd,'idx':np.arange(len(qq))}).sort_values('race_date')
    merged=pd.merge_asof(left,daily.sort_values('race_date'),on='race_date',by='_k',allow_exact_matches=False).sort_values('idx').set_index('idx').reindex(range(len(qq)))
    starts=merged['_start'].fillna(0.0)
    return pd.DataFrame({
        prefix+'_starts':np.log1p(starts),
        prefix+'_top5':(merged['_top5'].fillna(0.0)+alpha_top)/(starts+alpha_top+beta_top),
        prefix+'_win':(merged['_win'].fillna(0.0)+alpha_win)/(starts+alpha_win+beta_win),
    })

def prepare_fundamental(h, qq):
    qq=qq.copy().reset_index(drop=True)
    b=prepare(h,qq).reset_index(drop=True)
    out=b.drop(columns=['log_odds','market_prob','odds_missing']).copy()
    grp=qq.race_id.astype(str)

    # Musique réellement disponible avant course.
    music=qq.recent_form.apply(music_score)
    out['music_score']=pd.to_numeric(music,errors='coerce').fillna(0.45)
    out['music_missing']=pd.to_numeric(music,errors='coerce').isna().astype(float)
    out['music_rankpct']=_rankpct(out['music_score'],grp,ascending=False)

    # Variables relatives physiques, uniquement pré-course.
    raw_weight=pd.to_numeric(qq.weight,errors='coerce')
    raw_draw=pd.to_numeric(qq.draw,errors='coerce')
    out['light_weight_rankpct']=_rankpct(-raw_weight.fillna(raw_weight.groupby(grp).transform('median')),grp,ascending=False)
    out['inner_draw_rankpct']=_rankpct(-raw_draw.fillna(raw_draw.groupby(grp).transform('median')),grp,ascending=False)
    out['weight_spread']=raw_weight.groupby(grp).transform(lambda s: s.max()-s.min()).fillna(0.0)/10.0
    out['draw_distance_interaction']=out['inner_draw_rankpct'] * (1/(1+np.maximum(out['distance']-1.6,0))) * out.get('discipline_PLAT',0)
    out['light_distance_interaction']=out['light_weight_rankpct'] * np.clip(out['distance']/2.4,0.5,1.5) * out.get('discipline_PLAT',0)

    # Relative ranks des signaux fondamentaux historiques.
    for col in ['form','consistency','horse_name_top5','horse_name_win','jockey_top5','jockey_win','trainer_top5','trainer_win']:
        out[col+'_rankpct']=_rankpct(out[col],grp,ascending=False)
        out[col+'_rel']=out[col]-out[col].groupby(grp).transform('mean')

    # Jours depuis la dernière course (strictement antérieure).
    hh=h.copy(); hh['race_date']=pd.to_datetime(hh.race_date).dt.normalize(); hh['_ent']=key(hh.horse_name)
    last=(hh[hh['_ent'].ne('')][['_ent','race_date']].drop_duplicates().sort_values('race_date'))
    left=pd.DataFrame({'_ent':key(qq.horse_name),'race_date':pd.to_datetime(qq.race_date).dt.normalize(),'idx':np.arange(len(qq))}).sort_values('race_date')
    if len(last):
        m=pd.merge_asof(left,last.rename(columns={'race_date':'last_date'}).sort_values('last_date'),left_on='race_date',right_on='last_date',by='_ent',allow_exact_matches=False,direction='backward').sort_values('idx').set_index('idx').reindex(range(len(qq)))
        days=(pd.to_datetime(qq.race_date).dt.normalize()-m.last_date.reset_index(drop=True)).dt.days
    else: days=pd.Series(np.nan,index=qq.index)
    out['days_since_run_log']=np.log1p(pd.to_numeric(days,errors='coerce').clip(lower=0).fillna(30))
    out['fresh_7d']=pd.to_numeric(days,errors='coerce').between(0,7).astype(float)
    out['fresh_30d']=pd.to_numeric(days,errors='coerce').between(8,30).astype(float)
    out['layoff_60d']=pd.to_numeric(days,errors='coerce').gt(60).fillna(False).astype(float)

    # Contextes catégoriels pré-course.
    h2=h.copy(); q2=qq.copy()
    h2['distance_bucket']=_dist_bucket(h2.distance); q2['distance_bucket']=_dist_bucket(q2.distance)
    h2['discipline_ctx']=h2.discipline.fillna('INCONNU').astype(str); q2['discipline_ctx']=q2.discipline.fillna('INCONNU').astype(str)
    h2['hippo_ctx']=h2.hippodrome.fillna('INCONNU').astype(str); q2['hippo_ctx']=q2.hippodrome.fillna('INCONNU').astype(str)
    h2['terrain_ctx']=h2.terrain.fillna('INCONNU').astype(str); q2['terrain_ctx']=q2.terrain.fillna('INCONNU').astype(str)
    priors=[
        ('horse_name','discipline_ctx','horse_disc'),('horse_name','distance_bucket','horse_dist'),
        ('horse_name','hippo_ctx','horse_hippo'),('horse_name','terrain_ctx','horse_terrain'),
        ('jockey','discipline_ctx','jockey_disc'),('trainer','discipline_ctx','trainer_disc')]
    for ent,ctx,pref in priors:
        p=_cum_prior(h2,q2,ent,ctx,pref).reset_index(drop=True)
        out=pd.concat([out.reset_index(drop=True),p],axis=1)
        out[pref+'_top5_rankpct']=_rankpct(out[pref+'_top5'],grp,ascending=False)
        out[pref+'_win_rankpct']=_rankpct(out[pref+'_win'],grp,ascending=False)

    # Accord forme / musique / expérience contextuelle.
    out['form_music_mean']=(out['form_rankpct']+out['music_rankpct'])/2
    out['entity_context_mean']=(out['horse_disc_top5_rankpct']+out['horse_dist_top5_rankpct']+out['jockey_disc_top5_rankpct']+out['trainer_disc_top5_rankpct'])/4
    out['fundamental_consensus']=(out['form_rankpct']+out['music_rankpct']+out['horse_name_top5_rankpct']+out['jockey_top5_rankpct']+out['trainer_top5_rankpct'])/5
    return out.replace([np.inf,-np.inf],0).fillna(0).astype(float)

