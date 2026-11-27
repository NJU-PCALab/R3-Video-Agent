"""Explicit isolated postproduction cues; source video packets are copied intact."""
from copy import deepcopy
from array import array
import sys
import json
from pathlib import Path

from .storage import Pending, Unavailable, read, sha, write

VERSION = 'sparse_post_score_v1'


def validate(cues, ordered_ids):
    if len(set(cues)) != len(cues) or any(cid not in ordered_ids for cid in cues):
        raise ValueError('Unknown or duplicated score cue')
    positions = sorted(ordered_ids.index(cid) for cid in cues)
    if any(b-a < 3 for a,b in zip(positions,positions[1:])):
        raise ValueError('At least two unscored clips must separate newly authored cues')


def apply(agent, cues):
    if not cues:
        return
    s, media = agent.s, agent.media
    validate(cues, [x['clip']['id'] for x in s.data['selected']])
    selected = {x['clip']['id']: x for x in s.data['selected']}
    if s.data.get('score_receipt'):
        for cid, receipt in s.data['score_receipt']['clips'].items():
            if sha(receipt['path']) != receipt['sha256'] or sha(receipt['audio_path']) != receipt['audio_sha256']:
                raise Pending('Existing score mix changed; do not regenerate or remix it')
        return
    s.check_stop()
    total = sum(selected[cid]['receipt']['seconds'] for cid in cues)
    spec = {'generator':'music','role':'score','kind':'audio','seconds':total,
            'prompt':'Instrumental ancient Chinese period-drama underscore, one cohesive sparse motif. Soft breathy bamboo flute and a few guqin plucks, gentle warm sustained texture, no drums, no dense rhythmic pulse, no vocals, no humming, no speech. Three understated phrases: curious mountain journey, slightly brighter return journey, warm humble ending. Long natural rests, restrained dynamic, clean gentle endings. Music only.'}
    asset = media.audio('authorized_sparse_score_01',spec,seed=73113)
    track = asset.get('full_path',asset['path'])
    if media.audio_info(track)['seconds'] + 1/32000 < total:
        raise Pending('Score source is shorter than authorized cue windows; do not shorten native audio',status='score_needs_review')
    check = agent.artifact('score/instrumental_check',lambda:agent.advisory('score/instrumental_check',
        'Listen to this actual instrumental cue source. Return {status:"observed|unknown",no_human_voice:true|false,instrumental_quality:"clear|defect|unknown",observation}. Check actual audible voice, glitches and abrupt noise. Do not infer PASS from the prompt.',
        {'purpose':'Sparse Chinese period-drama score source; actual audio only'},provider='gemini',
        media=[{'path':track,'label':'ACTUAL separate music stem'}],max_tokens=2000))
    if check.get('status')!='observed' or check.get('no_human_voice') is not True or check.get('instrumental_quality')!='clear':
        raise Pending('Separate music stem needs actual review; no mix released',status='score_needs_review')
    receipts={};offset=0.
    for cid in cues:
        s.check_stop();row=selected[cid];source=row['receipt'].get('original_receipt',row['receipt']);folder=s.run/'media/score'/cid;folder.mkdir(parents=True,exist_ok=True)
        receipt=read(folder/'receipt.json')
        if not receipt:
            duration=source['frames']/24;audio=folder/'scored_native.wav';output=folder/'scored.mp4';floating=folder/'mixed_float.wav'
            # Native foley/voice is retained as one input. Duck the separate music under it.
            graph=(f'[0:a]atrim=duration={duration},asetpts=PTS-STARTPTS,aresample=32000,aformat=channel_layouts=stereo,asplit=2[n][side];'
                   f'[1:a]atrim=start={offset}:duration={duration},asetpts=PTS-STARTPTS,aresample=32000,'
                   f'loudnorm=I=-30:TP=-3:LRA=7,aresample=32000,aformat=channel_layouts=stereo,afade=t=in:st=0:d=0.7,afade=t=out:st={max(0,duration-1.2)}:d=1.2[m];'
                   '[m][side]sidechaincompress=threshold=0.015:ratio=8:attack=10:release=300[duck];'
                   '[n][duck]amix=inputs=2:duration=first:dropout_transition=0,volume=2[out]')
            media.command([media.ffmpeg,'-v','error','-y','-i',source['audio_path'],'-i',track,'-filter_complex',graph,
                           '-map','[out]','-c:a','pcm_f32le','-ar','32000','-ac','2',floating])
            samples=array('f',media.command([media.ffmpeg,'-v','error','-i',floating,'-f','f32le','-ac','2','-ar','32000','-']))
            if sys.byteorder!='little':samples.byteswap()
            peak=max((abs(v) for v in samples),default=0.)
            if peak>=1:
                raise Pending('Music mix lacks headroom; preserve original dry track and resolve mix gain',status='score_needs_review')
            media.command([media.ffmpeg,'-v','error','-y','-i',floating,'-c:a','pcm_s16le','-ar','32000','-ac','2',audio])
            media.command([media.ffmpeg,'-v','error','-y','-i',source['path'],'-i',audio,'-map','0:v:0','-map','1:a:0',
                           '-c:v','copy','-c:a','aac','-ar','32000','-ac','2','-movflags','+faststart',output])
            def video_hash(path):
                return media.command([media.ffmpeg,'-v','error','-i',path,'-map','0:v:0','-c:v','copy','-f','hash','-hash','sha256','-']).decode().strip()
            before,after=video_hash(source['path']),video_hash(output)
            receipt=media.probe(output)
            if before!=after or receipt['frames']!=source['frames'] or receipt['fps']!=24:
                raise Unavailable('Score mix changed the original video stream or frame count')
            receipt.update(kind='video',audio_path=str(audio),audio_sha256=sha(audio),native_audio=True,
                           original_receipt=deepcopy(source),score_stem_sha256=sha(track),source_video_stream_hash=before,
                           output_video_stream_hash=after,score_version=VERSION,score_offset_s=offset,
                           native_audio_preserved_in_mix=True, pre_quantization_peak=peak, limiter_or_normalization_of_native_applied=False)
            write(folder/'receipt.json',receipt)
        if sha(receipt['path'])!=receipt['sha256'] or sha(receipt['audio_path'])!=receipt['audio_sha256']:
            raise Pending('Retained score receipt bytes changed')
        row['receipt']=receipt;row['postproduction_score']={'version':VERSION,'original_review_retained':True,'stem':asset,'source_native_audio':source['audio_path']}
        receipts[cid]=receipt;offset+=source['seconds'];s.save()
    s.data['score_receipt']={'version':VERSION,'clips':receipts,'source_stem':asset,'instrumental_check':check,
                             'only_authorized_audio_edits':list(cues),'other_original_music_unchanged':True}
    s.save()
