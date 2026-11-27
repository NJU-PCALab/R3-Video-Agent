"""One deterministic reuse-only boundary shared by proposals and physical workers."""
from .explicit_recovery import permission
from .storage import Pending

# These receipts authorize reuse of one retained voice, not another generation.
_REUSE_PERMISSIONS = ('retained_voice_draft', 'retained_calibration_reference')


def restriction(config, store, spec, reference=None):
    """Return the matching restriction; renaming a voice never changes its scope."""
    assets = store.data.get('assets', {})
    for field in _REUSE_PERMISSIONS:
        permit = permission(config, field)
        if permit is None:
            continue
        if (permit.get('run') != str(store.run.resolve()) or not permit.get('asset_id')
                or permit.get('new_generation_allowed') is not False):
            raise Pending('Reuse-only asset permission differs from its run/scope', status='needs_configuration')
        root = permit['asset_id']
        root_spec = assets.get(root, {}).get('spec', {})
        speakers = {x for x in (permit.get('speaker'), root_spec.get('speaker')) if x}
        speakers.update(role for role, value in config.get('generation_contract', {}).get('cast', {}).items()
                        if value.get('voice_asset_id') == root)
        if spec.get('role') == 'voice' and spec.get('speaker') in speakers:
            return {'permission_field': field, 'asset_id': root, 'reason': 'same restricted speaker'}
        # Walk existing registry provenance as well as a newly proposed spec. The
        # visited set bounds malformed cycles without adding any lock or retry.
        pending = [spec, reference or {}]
        seen = set()
        while pending:
            item = pending.pop()
            if not isinstance(item, dict):
                continue
            ident = item.get('id')
            if ident == root:
                return {'permission_field': field, 'asset_id': root, 'reason': 'restricted asset lineage'}
            parent_ids = [item.get('reference_asset_id'), item.get('effective_reference_asset_id'),
                          item.get('canonical_voice_asset_id'),
                          item.get('canonical_voice_authority', {}).get('canonical_voice_asset_id')]
            if root in parent_ids:
                return {'permission_field': field, 'asset_id': root, 'reason': 'restricted asset lineage'}
            hashes = {item.get('sha256'), item.get('full_sha256'),
                      item.get('provenance', {}).get('reference_sha256')}
            if any(value and value in hashes for value in (permit.get('media_sha256'), permit.get('full_media_sha256'))):
                return {'permission_field': field, 'asset_id': root, 'reason': 'restricted original media'}
            for cursor in [ident] + parent_ids:
                if cursor and cursor not in seen:
                    seen.add(cursor)
                    row = assets.get(cursor, {})
                    pending.append(row.get('spec', {}))
                    pending.extend(row.get('versions', []))
    return None


def generation_blocked(agent, ident, row):
    spec = {**row.get('spec', {}), 'id': ident}
    return restriction(agent.config, agent.s, spec) is not None


def assert_generation_allowed(config, store, spec, reference=None):
    blocked = restriction(config, store, spec, reference)
    if blocked:
        raise Pending('Reuse-only permission forbids new root/variant/clone media: ' + blocked['asset_id'],
                      status='asset_generation_forbidden')
