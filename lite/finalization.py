"""Verify a retained final delivery after an interrupted runner exit."""
from pathlib import Path

from . import duration_budget
from .storage import Pending, read, sha
from .worker_wait import same_process


TERMINAL = ('completed', 'completed_with_warnings')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def _verify_delivery(run, selected, delivery):
    frames = sum(duration_budget.receipt_frames(item['receipt']) for item in selected)
    check(duration_budget.receipt_frames(delivery) == frames and delivery['seconds'] == frames / 24
          and delivery['video_streams'] == 1 and delivery['audio_streams'] == 1
          and delivery['production_complete'] is True and delivery['native_audio'] is True,
          'Final delivery does not match technical frame and native audio requirements')
    final = run / 'delivery/final.mp4'
    check(Path(delivery['path']).resolve(strict=True) == final and sha(final) == delivery['sha256'],
          'Final bytes differ from the retained technical receipt')
    ordered = []
    media = (run / 'media').resolve(strict=True)
    for item in selected:
        receipt = item['receipt']
        video = Path(receipt['path']).resolve(strict=True)
        audio = Path(receipt['audio_path']).resolve(strict=True)
        video.relative_to(media)
        audio.relative_to(media)
        check(receipt['native_audio'] is True and sha(video) == receipt['sha256']
              and sha(audio) == receipt['audio_sha256'],
              'Selected video or native PCM differs from its original receipt')
        ordered.append({'clip_id': item['clip']['id'], 'path': receipt['path'], 'sha256': receipt['sha256']})
    check(delivery['source_clips'] == ordered, 'Final source receipt order differs')


def retain_assembly(run, selected):
    """Reuse an existing full assembly only if its technical receipt still binds every file."""
    run = Path(run).resolve()
    absent = object()
    try:
        delivery = read(run / 'delivery/technical.json', absent)
        if delivery is absent:
            return None
        _verify_delivery(run, selected, delivery)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise Pending('Retained assembly needs reconciliation: ' + str(exc),
                      status='needs_finalization_reconciliation') from exc
    return delivery


def retained(run, config, state, report):
    """Finalize exact previously validated bytes without invoking any worker."""
    run = Path(run).resolve()
    try:
        previous = read(run / 'process.json', required=True)
        check(isinstance(previous, dict) and type(previous.get('pid')) is int and previous['pid'] > 0
              and str(previous.get('start_ticks', '')).isdigit()
              and isinstance(previous.get('argv'), list) and bool(previous['argv']),
              'Original runner identity is incomplete')
        for path in (run / 'exits').glob('*.json'):
            exit_record = read(path, required=True)
            if (same_process(exit_record.get('process'), previous) and exit_record.get('returncode') == 0
                    and exit_record.get('status') in TERMINAL):
                raise Pending('Completed result is immutable; use an explicitly authorized derivative run',
                              status='completed_run_immutable')
        check(not config.get('selective_repair'), 'Selective repair finalization must use its own verification')
        delivery = report['delivery']
        check(delivery == read(run / 'delivery/technical.json', required=True),
              'Report delivery differs from its original technical receipt')
        check(state.get('delivery') is None or state['delivery'] == delivery, 'Saved delivery differs from the report')
        source = sha(run / 'source.txt')
        check(report['source_sha256'] == source and config.get('source_text_sha256', source) == source,
              'Report does not bind the retained story source')
        selected = state['selected']
        ids = [item['clip']['id'] for item in selected]
        check(ids and len(ids) == len(set(ids)) and ids == [clip['id'] for clip in state['plan']['clips']],
              'Selected shots do not exhaust the saved ordered plan')
        check(report['clips'] == selected and report['counts'] == state['counts'] and report['usage'] == state['usage'],
              'Report differs from retained selections or execution ledger')
        duration_budget.final(config, selected)
        check(report.get('technical_completion', True) is True, 'Report does not mark technical completion')
        _verify_delivery(run, selected, delivery)
        outcomes = [item['quality']['ok'] for item in selected]
        check(all(type(value) is bool for value in outcomes), 'Retained content outcomes must be boolean')
        accepted = all(outcomes)
        check(report['quality_status'] == ('OK' if accepted else 'draft/needs_review')
              and report.get('content_acceptance_passed', accepted) is accepted,
              'Reported content acceptance differs from retained shot reviews')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise Pending('Retained final delivery needs reconciliation: ' + str(exc),
                      status='needs_finalization_reconciliation') from exc
    return delivery, 'completed' if accepted else 'completed_with_warnings', previous
