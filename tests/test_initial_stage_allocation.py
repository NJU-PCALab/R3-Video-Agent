"""Replay observed initial-wire shapes without a model or generated media."""
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase

from lite import character_stages as stages
from lite.storage import Pending


def fixture():
    config = {'character_reference_policy':{'version':stages.VERSION,
        'max_stages_per_character':3,'max_candidates_per_character':2,
        'require_stage_declarations':True,'allocation_authority':'host_family_cap_v1'},
        'generation_contract':{'cast':{'Alice':{'image_asset_id':'char_r1','voice_asset_id':'voice_r1'}}}}
    plan = {'assets':[{'id':'char_r1','kind':'image','role':'character','prompt':'Young French woman.'}],
        'character_reference_stages':{'Alice':{'stages':[
            {'id':'young','image_asset_id':'char_r1','description':'Young adult woman.','candidate_limit':2},
            {'id':'aged','image_asset_id':'char_r1_aged','description':'Same woman after ten years of hard labour.','candidate_limit':2}]}},
        'clips':[{'id':'clip_001','seconds':8,'asset_ids':['char_r1'],'dialogue':[{'speaker':'Alice','text':'I would love to go.'}],
                  'character_states':{'Alice':{'stage':'young'}}},
                 {'id':'clip_020','seconds':8,'asset_ids':['char_r1'],'dialogue':[{'speaker':'Alice','text':'We worked for ten years.'}],
                  'character_states':{'Alice':{'stage':'aged'}}}]}
    return plan,config


class InitialStageAllocationTests(TestCase):
    def test_role_family_list_keeps_raw_stage_rows_and_rejects_duplicates(self):
        original,config=fixture()
        raw=[{'canonical_role':'Alice',**original['character_reference_stages']['Alice']}]
        original['character_reference_stages']=raw
        result=stages.prepare_plan(original,config)
        self.assertEqual(result['character_stage_allocation_receipt']['proposed_table'],raw)
        self.assertEqual(result['character_stage_allocation_receipt']['retained_table_shape_normalization'],
                         'canonical_role_list_to_role_keyed_table')
        self.assertEqual(stages.prepare_plan(result,config),result)
        with self.assertRaisesRegex(Pending,'at most once'):
            stages.prepare_plan({**original,'character_reference_stages':raw+raw},config)

    def test_only_host_declared_stable_roles_can_reuse_their_canonical_anchor(self):
        original,config=fixture();cast=config['generation_contract']['cast']
        cast['Infant']={'image_asset_id':'infant','voice_asset_id':None,'description':'The same swaddled infant.'}
        original['assets'].append({'id':'infant','kind':'image','role':'character','prompt':'Same sleeping swaddled infant.'})
        original['clips'][0]['asset_ids'].append('infant')
        with self.assertRaisesRegex(Pending,'every canonical cast role'):
            stages.prepare_plan(original,config)
        config['character_reference_policy']['stable_single_stage_roles']=['Infant']
        result=stages.prepare_plan(original,config)
        self.assertEqual(result['clips'][0]['character_stage_bindings']['Infant']['image_asset_id'],'infant')
        self.assertEqual(result['character_stage_allocation_receipt']['host_declared_stable_single_stage_roles'],['Infant'])
        self.assertEqual(stages.prepare_plan(result,config),result)
        original['clips'][0]['character_states']['Infant']={'stage':'unknown'}
        with self.assertRaisesRegex(Pending,'no explicit stage selection'):
            stages.prepare_plan(original,config)

    def test_first_stage_can_use_exact_canonical_image_only_for_opted_in_role(self):
        original,config=fixture();del original['clips'][0]['character_states']
        with self.assertRaisesRegex(Pending,'First visible'):
            stages.prepare_plan(original,config)
        config['character_reference_policy']['first_stage_from_bound_image_roles']=['Alice']
        result=stages.prepare_plan(original,config)
        self.assertEqual(result['clips'][0]['character_stage_bindings']['Alice']['stage'],'young')
        self.assertEqual(result['clips'][1]['character_stage_bindings']['Alice']['stage'],'aged')
        self.assertEqual(stages.prepare_plan(result,config),result)
        original['clips'][0]['asset_ids'].append('char_r1_aged')
        with self.assertRaisesRegex(Pending,'First visible'):
            stages.prepare_plan(original,config)

    def test_retained_role_stage_list_keeps_rows_and_original_wire_audit(self):
        original,config=fixture()
        original['character_reference_stages']['Alice']=original['character_reference_stages']['Alice']['stages']
        before=deepcopy(original)
        result=stages.prepare_plan(original,config)
        self.assertEqual(original,before)
        self.assertEqual(result['character_stage_allocation_receipt']['proposed_table'],before['character_reference_stages'])
        self.assertEqual(result['character_stage_allocation_receipt']['retained_family_shape_normalizations'][0]['role'],'Alice')
        self.assertEqual([s['image_asset_id'] for s in result['character_reference_stages']['Alice']['stages']],['char_r1','char_r1_aged'])
        self.assertEqual([s['candidate_limit'] for s in result['character_reference_stages']['Alice']['stages']],[1,1])
        self.assertEqual([c['dialogue'] for c in result['clips']],[c['dialogue'] for c in before['clips']])
        self.assertEqual(stages.prepare_plan(result,config),result)

    def test_list_shape_does_not_infer_missing_image_ids_or_unknown_roles(self):
        for mode in ('unknown_role','empty','missing_image','duplicate_image','invalid_rows'):
            original,config=fixture();rows=original['character_reference_stages']['Alice']['stages']
            original['character_reference_stages']['Alice']=rows
            if mode=='unknown_role':original['character_reference_stages']['not_cast']=original['character_reference_stages'].pop('Alice')
            elif mode=='empty':rows.clear()
            elif mode=='missing_image':rows[0]['canonical_image_asset_id']=rows[0].pop('image_asset_id')
            elif mode=='duplicate_image':rows[1]['image_asset_id']=rows[0]['image_asset_id']
            else:rows[0]='not a stage'
            with self.subTest(mode=mode),self.assertRaises(Pending):stages.prepare_plan(original,config)

    def test_list_shape_does_not_authorize_old_policy_or_age_regression(self):
        original,config=fixture();original['character_reference_stages']['Alice']=original['character_reference_stages']['Alice']['stages']
        strict=deepcopy(config);strict['character_reference_policy'].pop('allocation_authority')
        with self.assertRaises(Pending):stages.prepare_plan(original,strict)
        original['clips'].append(deepcopy(original['clips'][0]));original['clips'][-1]['id']='clip_021'
        with self.assertRaisesRegex(Pending,'regressed'):stages.prepare_plan(original,config)

    def test_two_stage_families_two_plus_two_becomes_one_plus_one_without_source_rewrite(self):
        original,config=fixture();before=deepcopy(original)
        normalized=stages.prepare_plan(original,config)
        self.assertEqual(original,before)
        actual=normalized['character_reference_stages']['Alice']['stages']
        self.assertEqual([x['candidate_limit'] for x in actual],[1,1])
        self.assertEqual([x['image_asset_id'] for x in actual],['char_r1','char_r1_aged'])
        self.assertEqual([c['dialogue'] for c in normalized['clips']],[c['dialogue'] for c in original['clips']])
        self.assertEqual([c['id'] for c in normalized['clips']],[c['id'] for c in original['clips']])
        audit=normalized['character_stage_allocation_receipt']
        self.assertEqual(audit['proposed_table'],before['character_reference_stages'])
        self.assertEqual([x['candidate_limit'] for x in audit['effective']['Alice']['stages']],[1,1])
        self.assertEqual(audit['version'],'host_family_cap_v1')
        self.assertFalse(audit['budget_increased'])
        self.assertEqual(stages.prepare_plan(normalized,config),normalized)

    def test_old_configuration_retains_strict_two_plus_two_rejection(self):
        original,config=fixture();config['character_reference_policy'].pop('allocation_authority')
        with self.assertRaisesRegex(Pending,'shared'):stages.prepare_plan(original,config)

    def test_canonical_asset_id_cannot_be_shared_between_life_stages(self):
        original,config=fixture();original['character_reference_stages']['Alice']['stages'][1]['image_asset_id']='char_r1'
        with self.assertRaises(Pending):stages.prepare_plan(original,config)

    def test_three_stages_do_not_fit_two_slots_but_explicit_family_three_is_bounded(self):
        original,config=fixture();original['character_reference_stages']['Alice']['stages'].append(
            {'id':'elder','image_asset_id':'char_r1_elder','description':'Same woman in late life.','candidate_limit':2})
        with self.assertRaisesRegex(Pending,'stage count'):stages.prepare_plan(original,config)
        config['character_reference_policy']['max_candidates_per_character']=3
        normalized=stages.prepare_plan(original,config)
        self.assertEqual([x['candidate_limit'] for x in normalized['character_reference_stages']['Alice']['stages']],[1,1,1])

    def test_stable_single_stage_keeps_original_two_slots(self):
        original,config=fixture();original['character_reference_stages']['Alice']['stages'].pop()
        original['clips'].pop()
        normalized=stages.prepare_plan(original,config)
        self.assertEqual(normalized['character_reference_stages']['Alice']['stages'][0]['candidate_limit'],2)

    def test_exact_time_jump_annotation_is_retained_and_other_unknown_roles_rejected(self):
        original,config=fixture();note='Ten years pass; this note is not a cast role.'
        original['character_reference_stages']['time_jump_note']=note
        normalized=stages.prepare_plan(original,config)
        self.assertNotIn('time_jump_note',normalized['character_reference_stages'])
        self.assertEqual(normalized['character_reference_stage_notes'],[
            {'source_field':'character_reference_stages.time_jump_note','value':note}])
        self.assertEqual(normalized['character_stage_allocation_receipt']['proposed_table']['time_jump_note'],note)
        self.assertEqual(stages.prepare_plan(normalized,config),normalized)
        original['character_reference_stages']['imagined_role']={'stages':[]}
        with self.assertRaisesRegex(Pending,'canonical cast role'):stages.prepare_plan(original,config)

    def test_non_string_annotation_is_not_silently_treated_as_metadata(self):
        original,config=fixture();original['character_reference_stages']['time_jump_note']={'stages':[]}
        with self.assertRaises(Pending):stages.prepare_plan(original,config)

    def test_allocation_and_original_proposal_audit_are_immutable_after_freeze(self):
        original,config=fixture();normalized=stages.prepare_plan(original,config)
        store=SimpleNamespace(data={},save=lambda:None);receipt=stages.freeze_plan(store,normalized,config)
        self.assertEqual(receipt['allocation_audit'],normalized['character_stage_allocation_receipt'])
        changed=deepcopy(normalized);changed['character_stage_allocation_receipt']['proposed_table']['Alice']['stages'][0]['candidate_limit']=1
        with self.assertRaises(Pending):stages.freeze_plan(store,changed,config)
        changed=deepcopy(normalized);changed['character_reference_stages']['Alice']['stages'][0]['candidate_limit']=2
        with self.assertRaises(Pending):stages.prepare_plan(changed,config)

    def test_explicit_reversible_appearance_can_return_clean_without_authorizing_age_regression(self):
        original,config=fixture();family=original['character_reference_stages'].pop('Alice')
        config['generation_contract']['cast']={'马骥':{'image_asset_id':'char_r1','voice_asset_id':'voice_r1'}}
        original['character_reference_stages']['马骥']=family
        for row,ident,description in zip(family['stages'],['plain','coal'],['Same adult with clean face.','Same adult with removable coal face paint.']):
            row.update(id=ident,description=description)
        for c,stage in zip(original['clips'],['plain','coal']):
            c['dialogue']=[];c['character_states']={'马骥':{'stage':stage}}
        original['clips'].append({'id':'clip_017','seconds':8,'asset_ids':['char_r1'],'dialogue':[],
                                  'character_states':{'马骥':{'stage':'plain'}}})
        with self.assertRaisesRegex(Pending,'regressed'):stages.prepare_plan(original,config)
        config['character_reference_policy']['role_transition_modes']={'马骥':'reversible_appearance'}
        normalized=stages.prepare_plan(original,config)
        self.assertEqual([c['character_stage_bindings']['马骥']['stage'] for c in normalized['clips']],['plain','coal','plain'])
        store=SimpleNamespace(data={},save=lambda:None);receipt=stages.freeze_plan(store,normalized,config)
        self.assertEqual(receipt['role_transition_modes'],{'马骥':'reversible_appearance'})
        different=deepcopy(config);different['character_reference_policy']['role_transition_modes']={}
        with self.assertRaises(Pending):stages.freeze_plan(store,normalized,different)
        age,cfg=fixture();cfg['character_reference_policy']['role_transition_modes']={'Alice':'chronological_age'}
        age['clips'].append(deepcopy(age['clips'][0]));age['clips'][-1]['id']='clip_021'
        with self.assertRaisesRegex(Pending,'regressed'):stages.prepare_plan(age,cfg)

    def test_role_transition_policy_requires_explicit_known_role_and_finite_mode(self):
        for modes in ({'unknown':'reversible_appearance'},{'Alice':'any_change'},['Alice']):
            original,config=fixture();config['character_reference_policy']['role_transition_modes']=modes
            with self.subTest(modes=modes),self.assertRaises(Pending):stages.prepare_plan(original,config)
