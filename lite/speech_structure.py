"""Use remaining format slots of the original same-media clarification only."""
import hashlib,json
from .storage import read,write,Pending,Unavailable

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def validate(report,disagrees):
    if disagrees and isinstance(report,dict) and isinstance(report.get('asr_discrepancy'),dict) and report['asr_discrepancy'].get('classification')=='none':
        raise ValueError('Independent ASR text differs from the source. classification none contradicts this known difference; explicitly classify actual_defect, asr_error, homophone_transcription or unknown using the same actual media, preserving all real defects and uncertainties.')
    return report

def reconcile(agent,operation,prompt,payload,media,report,contract,cid,take):
    try:return validate(report,contract['asr_text_disagrees'])
    except ValueError:pass
    binding={'media_sha256':contract['media_sha256'],'audio_sha256':contract['audio_sha256'],'original_report_sha256':digest(report),'blind_asr_sha256':digest(contract['blind_asr']),'operation':operation}
    path=agent.s.run/'review_media'/cid/str(take)/'speech_structure_recovery.json';saved=read(path)
    if saved is not None:
        if saved.get('binding')!=binding:raise Pending('Original same-media speech structure receipt changed; reconcile without new observation',status='needs_speech_observation_clarification')
        return validate(saved['report'],True)
    try:
        corrected=agent.models.json(operation,prompt,payload,provider='gemini',media=media,max_tokens=5000,validate=lambda x:validate(x,True))
    except Unavailable as exc:
        raise Pending('Original same-media clarification format slots remain unresolved: '+str(exc),status='needs_speech_observation_clarification') from exc
    write(path,{'binding':binding,'report':corrected,'original_paid_report_retained':True,'same_logical_operation':operation,'observation_rounds':1,'maximum_original_format_slots':3,'not_media_improvement':True})
    return corrected
