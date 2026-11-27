"""Lossless partition adapter for an explicitly segmented scripted occurrence."""

def groups(lines, timeline, authority, *, allow_partition):
    count=len(lines)
    if not isinstance(timeline,list):
        raise ValueError('Dialogue timeline must be a list')
    if len(timeline)==count:
        return [[row] for row in timeline]
    if isinstance(authority,list) and len(authority)==count:
        return [[row] for row in authority]
    if not allow_partition or len(timeline)<=count:
        raise ValueError('Every scripted occurrence needs an explicit on/off-screen timeline')
    result=[];cursor=0
    for line in lines:
        target=line['text'].strip();assembled='';parts=[]
        while cursor<len(timeline) and assembled!=target:
            row=timeline[cursor]
            if (not isinstance(row,dict) or row.get('speaker')!=line['speaker']
                    or not isinstance(row.get('text'),str) or not row['text']):
                raise ValueError('Partition must explicitly preserve the source speaker and text')
            assembled+=row['text']
            if not target.startswith(assembled):
                raise ValueError('Partition text must exactly reconstruct one source occurrence')
            parts.append(row);cursor+=1
        if assembled!=target:
            raise ValueError('Partition omitted source dialogue')
        result.append(parts)
    if cursor!=len(timeline):
        raise ValueError('Partition added or repeated source dialogue')
    return result


def visibility(value):
    """Canonicalize only finite exact spelling aliases, never infer visibility."""
    aliases={'on':'on_screen','off':'off_screen','on_screen':'on_screen','on-screen':'on_screen','onscreen':'on_screen',
             'off_screen':'off_screen','off-screen':'off_screen','offscreen':'off_screen'}
    try:
        return aliases[value]
    except (KeyError,TypeError):
        raise ValueError('Unknown speaking visibility: use on_screen or off_screen for each dialogue occurrence') from None

def project_visibility(occurrences):
    """Expand explicit shorthand before storing source dialogue and its timeline."""
    if not isinstance(occurrences, list):
        return occurrences
    result = []
    for row in occurrences:
        if isinstance(row, dict) and row.get('on_screen_or_offscreen') in ('on', 'off'):
            row = {**row, 'on_screen_or_offscreen': visibility(row['on_screen_or_offscreen'])}
        result.append(row)
    return result
