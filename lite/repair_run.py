"""A bounded derivative cut: regenerate explicit targets, freeze the parent film."""
from copy import deepcopy
from pathlib import Path

from . import speech_contract, sparse_score
from .storage import Pending, candidate_quality, read, sha, write


def forward_boundaries(agent, targets):
    from .review import visual_frames
    selected=agent.s.data['selected'];records=[]
    for index,row in enumerate(selected[:-1]):
        if row['clip']['id'] not in targets:
            continue
        following=selected[index+1];name='repair_forward_boundary/'+row['clip']['id']
        media=[]
        for label,item in [('PREVIOUS_REGENERATED',row),('CURRENT_FROZEN',following)]:
            frames=visual_frames(agent.config,agent.s.run,item['receipt'],name.replace('/','_')+'_'+label)
            media.extend({'path':x['path'],'label':f'{label} {x["time_s"]:.3f}s'} for x in frames['frames'])
        review=agent.artifact(name,lambda:agent.advisory(name,
            '检查这两个完整4fps无声序列的实际切点、人物身份、服装、场景空间与叙事状态衔接。硬切换机位景别不等于不连续。不要判断声音，不要把4fps无法排除未采到帧异常记为缺陷。只报告实际观察到的跨切点冲突，不能以摄影计划代替事实。返回 {status:"observed|unknown",boundary_status:"clear|defect|unknown",observation,findings:[{category,observation,time_s}]}。',
            {'previous_seconds':row['receipt']['seconds'],'current_seconds':following['receipt']['seconds'],
             'scope':'Boundary compatibility only, not whole-film review'},provider='doubao',media=media,max_tokens=3500))
        record={'previous_id':row['clip']['id'],'current_id':following['clip']['id'],
                'previous_sha256':row['receipt']['sha256'],'current_sha256':following['receipt']['sha256'],'review':review}
        records.append(record);write(agent.s.run/'repair_forward_boundaries.json',records)
        if review.get('status')!='observed' or review.get('boundary_status')!='clear':
            raise Pending('New clip forward boundary requires actual resolution: '+row['clip']['id'],status='needs_boundary_review')
    return records


def run(agent):
    s=agent.s;cfg=agent.config['selective_repair'];parent=Path(cfg['parent_run'])
    if sha(parent/'report.json')!=cfg['parent_report_sha256'] or sha(parent/'state.json')!=cfg['parent_state_sha256']:
        raise Pending('Frozen parent result changed; do not repair against a different film')
    original=read(parent/'report.json');ordered=original['clips'];targets=cfg['target_clips']
    if set(targets)!=set(cfg['replacement_sources']) or len(set(targets))!=len(targets):
        raise Pending('Selective repair source set differs from authorization')
    if set(targets)&set(cfg.get('score_only_clips',[])):
        raise Pending('A clip cannot be both regenerate and score-only')
    for row in ordered:
        s.check_stop();cid=row['clip']['id'];receipt=row['receipt']
        if sha(receipt['path'])!=receipt['sha256'] or sha(receipt['audio_path'])!=receipt['audio_sha256']:
            raise Pending('Parent media differs from frozen source receipt: '+cid)
        existing=next((x for x in s.data['selected'] if x['clip']['id']==cid),None)
        if existing:
            continue
        if len(s.data['selected'])!=[x['clip']['id'] for x in ordered].index(cid):
            raise Pending('Repair selected prefix is not in original order')
        if cid in targets:
            agent.render_clip(deepcopy(cfg['replacement_sources'][cid]))
        else:
            frozen=deepcopy(row);frozen['parent_media_frozen']=True
            s.data['selected'].append(frozen);s.save()
            s.event('parent_clip_reused',clip_id=cid,media_sha256=receipt['sha256'],audio_sha256=receipt['audio_sha256'])
    for row in s.data['selected']:
        if row['clip']['id'] in targets and not candidate_quality(next(x for x in s.data['candidates'][row['clip']['id']] if x['take']==row['take']))['ok']:
            raise Pending('Target repair still has open content findings',status='content_release_blocked')
    boundaries=forward_boundaries(agent,targets)
    sparse_score.apply(agent,cfg.get('score_only_clips',[]))
    mix_reviews={}
    for row in s.data['selected']:
        cid=row['clip']['id']
        if cid not in cfg.get('score_only_clips',[]):
            continue
        receipt=row['receipt'];name='score/final_mix/'+cid
        review=agent.artifact(name,lambda:agent.advisory(name,
            '核验这段实际完成的稀疏配乐混音。音乐是允许的，不是unexpected_music。只检查实际听到的音乐是否突起/截断、是否遮盖现场动作声/人体发声、是否引入歌词或额外对白，以及声音与当前画面的关联。无需想象未发生的动作音。返回 {status:"observed|unknown",mix_quality:"clear|defect|unknown",no_added_speech:true|false,native_events_audible:true|false,observation}。',
            {'music_allowed':True,'current_seconds':receipt['seconds']},provider='gemini',
            media=[{'path':receipt['path'],'label':'CURRENT actual scored video'},{'path':receipt['audio_path'],'label':'CURRENT same mixed WAV'}],max_tokens=2500))
        mix_reviews[cid]=review;write(s.run/'score_mix_reviews.json',mix_reviews)
        if review.get('status')!='observed' or review.get('mix_quality')!='clear' or review.get('no_added_speech') is not True or review.get('native_events_audible') is not True:
            raise Pending('Actual scored clip mix requires review: '+cid,status='score_needs_review')
    s.status('running',stage='repair.assemble')
    delivery=agent.media.assemble(s.data['selected'])
    frozen=[x['clip']['id'] for x in ordered if x['clip']['id'] not in targets+cfg.get('score_only_clips',[])]
    inherited=[x['clip']['id'] for x in s.data['selected'] if x['clip']['id'] not in targets and not x['quality']['ok']]
    report={'delivery':delivery,'clips':s.data['selected'],'quality_status':'targeted_repairs_verified_with_inherited_warnings' if inherited else 'OK',
            'targeted_repairs':targets,'targeted_repairs_passed':True,'inherited_needs_review':inherited,
            'unchanged_av_clips':frozen,'score_only_clips':cfg.get('score_only_clips',[]),
            'score_receipt':s.data.get('score_receipt'),'score_mix_reviews':mix_reviews,'forward_boundary_reviews':boundaries,'parent_result':cfg,
            'counts':s.data['counts'],'usage':s.data['usage'],'parent_counts':original['counts'],'parent_usage':original['usage'],
            'currency_cost':None,'cost_note':'New repair usage separate from preserved parent usage; unknown currency is null.',
            'framework_release':agent.config['framework_release'],'local_content_rsi':s.data['local_lessons'],
            'global_content_rsi':'disabled_for_frozen_parent_repair','whole_film_review':'not_requested','source_code_rsi':False,
            'warnings':s.data['warnings'],'source_sha256':sha(s.run/'source.txt')}
    write(s.run/'report.json',report);s.data['delivery']=delivery
    s.status('completed_with_warnings' if inherited else 'completed',stage='repair.finished',path=delivery['path'])
    return report
