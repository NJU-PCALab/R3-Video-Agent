"""Regress the actual H3 direction and source targets after an accepted future edit."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from lite import binding_contract, completion_flow as flow, shot_policy


def spec(action):
    return {'opening':'The brush touches the paper.', 'action':action,
            'camera':'Hold a close angle.', 'closing':'The brush remains motionless.'}


class EffectiveReplanTests(unittest.TestCase):
    def setUp(self):
        self.old = 'He audibly murmurs a long string of official titles.'
        self.new = 'His lips move silently as the brush trembles.'
        self.clip = {'id':'clip_011', 'seconds':9, 'scene':'desk', 'action':self.old,
                     'action_beats':[{'action':self.old, 'visible_result':'He finishes speaking.'}],
                     'shot_design':{'opening':self.old, 'closing':'He finishes speaking.', 'cut_reason':self.old},
                     'microshots':[{'action':self.old}], 'parent_event_action':self.old,
                     'render_spec':spec(self.new), 'entry_state':self.old,
                     'exit_state':'Silent at the desk.', 'asset_ids':[], 'dialogue':[],
                     'state_checkpoints':{'brush':{'entry':'The brush is inked.', 'exit':'The brush remains inked.'}}}
        self.checkpoints = {}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.agent = SimpleNamespace(
            s=SimpleNamespace(data={'selected':[],'candidates':{},'replan_receipts':self.checkpoints},
                              run=Path(self.tmp.name),warning=Mock()),
            config={'completion_policy':{'version':flow.VERSION},'production_contract':{'version':shot_policy.CONTRACT_VERSION},
                    'generation_contract':{'version':binding_contract.VERSION,'cast':{'Yan':{'image_asset_id':'yan','voice_asset_id':None}}}},
            catalog=lambda:[], advisory=Mock(return_value={'render_spec':spec(self.old),'framing_emphasis':'result'}),
            artifact=lambda _name, function:function())
        self.record(3, render_spec=spec(self.new), exit_state=self.clip['exit_state'])

    def record(self, checkpoint, status='applied', **changes):
        self.checkpoints[str(checkpoint)] = {'receipts':[{'clip_id':'clip_011', 'status':status, 'changes':changes}]}

    def test_new_local_input_and_actual_h3_caption_use_approved_staging(self):
        source = deepcopy(self.clip)
        proposed = flow.generation_plan(self.agent,self.clip,1,'local_011_1',[])
        payload = self.agent.advisory.call_args.args[2]['current_event']
        self.assertNotIn(self.old, json.dumps(payload))
        self.assertEqual(payload['state_checkpoints'],source['state_checkpoints'])
        approved = flow.execution_clip(self.agent,self.clip)
        bound = flow.compile_plan(self.agent,self.clip,proposed)
        self.assertEqual(bound['render_spec']['action'],self.new)
        # Also exercise the lower-level correction/legacy-plan path independently.
        stale = {**bound,'render_spec':spec(self.old)}
        caption,binding = binding_contract.compile(approved,[],self.agent.config,stale)
        self.assertNotIn(self.old,caption)
        self.assertNotIn('finishes speaking',caption)
        self.assertIn('Beat 1: the brush trembles.',caption)
        self.assertNotIn('lips move',caption)
        self.assertEqual(caption.count('Keep the final physical result unobstructed.'),1)
        self.assertIn('The brush remains inked.',caption)
        self.assertEqual(binding['source_action_state_targets']['action_beats'],approved['action_beats'])
        self.assertEqual(self.clip, source)

    def test_replayed_legacy_local_artifact_cannot_restore_old_directions(self):
        saved = self.agent.s.run/'artifacts/local_011_1.json'
        saved.parent.mkdir()
        original = json.dumps({'decision':'generate','render_spec':spec(self.old),'shot_design':{'opening':self.old}})
        saved.write_text(original)
        proposed = flow.generation_plan(self.agent,self.clip,1,'local_011_1',[])
        bound = flow.compile_plan(self.agent,self.clip,proposed)
        self.assertEqual(bound['render_spec'],self.clip['render_spec'])
        self.assertNotIn('shot_design',bound)
        self.assertEqual(saved.read_text(),original)
        self.agent.advisory.assert_not_called()

    def test_receipts_merge_in_numeric_order_and_ignore_rejected_edits(self):
        self.record(20,status='rejected',render_spec=spec(self.old))
        self.record(15, exit_state='The lamp goes dark.')
        self.record(6, entry_state='The paper is blank.')
        new = {**self.clip['render_spec'], 'camera':'Frame the hands.'}
        self.record(10, render_spec=new)
        self.clip['render_spec'] = new
        result = flow.execution_clip(self.agent,self.clip)
        self.assertEqual(result['entry_state'],'The paper is blank.')
        self.assertEqual(result['exit_state'],'The lamp goes dark.')
        self.assertIn('Frame the hands.',result['shot_design']['camera'])

    def test_split_or_other_shot_never_acquires_parent_direction(self):
        for changed in ({'id':'clip_012'}, {'render_spec':spec('Only the assigned first part.'),
                        'completion_split':{'parent_clip_id':'clip_011'}}):
            partial = {**self.clip,**changed}
            self.assertEqual(flow.execution_clip(self.agent,partial),partial)
            self.assertEqual(shot_policy.execution_direction(partial, {'render_spec':spec('Local action.')})['action'],'Local action.')

    def test_protected_dialogue_and_role_identity_remain_exact(self):
        lines = [{'speaker':'Yan','text':'I will write.','on_screen_or_offscreen':'off_screen'}]
        self.clip.update(dialogue=lines,character_states={'Yan':{'stage':'adult'}},
                         asset_ids=['hat'],dialogue_timeline=[{'start_s':.7,'end_s':2.5}])
        result = flow.execution_clip(self.agent,self.clip)
        for k in ('dialogue','asset_ids','dialogue_timeline','character_states','seconds','scene','id'):
            self.assertEqual(self.clip[k],result[k])

    def test_reference_only_or_unapplied_patch_preserves_beats(self):
        self.checkpoints.clear()
        self.record(3,status='noop',render_spec=spec(self.new))
        self.record(6,asset_ids=['lamp'])
        self.assertIs(flow.execution_clip(self.agent,self.clip),self.clip)

    def test_endpoint_only_edit_keeps_independent_action_and_local_camera(self):
        self.checkpoints.clear()
        self.record(3,entry_state='The candle is already lit.')
        self.clip['entry_state'] = 'The candle is already lit.'
        view = flow.execution_clip(self.agent,self.clip)
        self.assertEqual(view['action_beats'],self.clip['action_beats'])
        self.assertEqual(view['state_checkpoints'],self.clip['state_checkpoints'])
        direction = shot_policy.execution_direction(view,{'render_spec':spec('Inspect the wick.')})
        self.assertEqual(direction['opening'],'The candle is already lit.')
        self.assertEqual(direction['action'],'Inspect the wick.')
        self.assertEqual(direction['closing'],spec('')['closing'])

    def test_post_replan_exit_only_updates_both_h3_blocks(self):
        self.record(6,exit_state='The paper slips off the desk.')
        self.clip['exit_state']='The paper slips off the desk.'
        view=flow.execution_clip(self.agent,self.clip)
        self.assertEqual(flow.execution_clip(self.agent,view),view)
        local=flow.compile_plan(self.agent,view,{'render_spec':spec(self.old)})
        caption,_=binding_contract.compile(view,[],self.agent.config,local)
        self.assertNotIn(spec(self.new)['closing'],caption)
        self.assertIn('The paper slips off the desk.',caption)
        self.assertEqual(view['action_beats'][0]['visible_result'],view['exit_state'])

    def make_prepared(self):
        from lite.media import video_request
        plan = {'render_spec':spec(self.old),'dialogue_timeline':[]}
        old_caption,binding=binding_contract.compile(self.clip,[],self.agent.config,plan)
        c = {'source_clip':deepcopy(self.clip),'plan':plan,'take':1,'references':[],
             'caption':old_caption,'binding_contract':binding}
        folder=self.agent.s.run/'media/videos/clip_011_take_1'
        self.agent.s.save=Mock()
        self.agent.s.data['candidates']={'clip_011':[c]}
        return c,folder,video_request(old_caption,9,[],11101),flow.execution_clip(self.agent,self.clip)

    def test_prepared_candidate_before_request_is_updated_without_dispatch(self):
        from lite.storage import read
        c,folder,_,clip = self.make_prepared()
        before=deepcopy(c)
        flow.reconcile_prepared_direction(self.agent,clip,c)
        self.assertFalse(folder.exists())
        self.assertEqual(read(Path(c['replan_pre_dispatch_evidence']))['original_candidate'],before)
        self.assertNotIn(self.old,c['caption'])
        self.assertEqual(c['take'],1)
        flow.reconcile_prepared_direction(self.agent,clip,c)
        self.agent.s.save.assert_called_once()

    def test_queued_pre_submission_request_is_repaired_in_place_and_replayable(self):
        from lite.storage import read,write
        c,folder,request,clip=self.make_prepared()
        before=deepcopy(c)
        write(folder/'request.json',request)
        write(folder/'cast_binding_contract.json',c['binding_contract'])
        write(folder/'resource_queue.json',{'status':'waiting','new_submission':False})
        flow.reconcile_prepared_direction(self.agent,clip,c)
        self.assertEqual(read(folder/'request.json'),{**request,'prompt':c['caption']})
        self.assertEqual(read(folder/'cast_binding_contract.json'),c['binding_contract'])
        audit=read(Path(c['replan_pre_dispatch_evidence']))
        self.assertEqual(audit['request'],request)
        self.assertEqual(audit['binding'],before['binding_contract'])
        # Simulate shutdown after replacing the disk request but before saving state.
        old_candidate=deepcopy(before)
        flow.reconcile_prepared_direction(self.agent,clip,old_candidate)
        self.assertEqual(old_candidate,c)
        self.assertFalse((folder/'submitted.json').exists())

    def test_paid_or_ambiguous_attempt_is_never_rebound(self):
        from lite.storage import Pending,write
        c,folder,request,clip=self.make_prepared()
        before=deepcopy(c)
        self.agent.s.data['accounted_operations']=[str(folder)]
        flow.reconcile_prepared_direction(self.agent,clip,c)
        self.assertEqual(c,before)
        self.agent.s.data['accounted_operations']=[]
        write(folder/'process.json',{'pid':17})
        with self.assertRaises(Pending):
            flow.reconcile_prepared_direction(self.agent,clip,c)
        self.assertEqual(c,before)
        write(folder/'submitted.json',{'time':0})
        flow.reconcile_prepared_direction(self.agent,clip,c)
        self.assertEqual(c,before)
        self.agent.s.save.assert_not_called()

    def test_different_queued_seed_is_never_replaced(self):
        from lite.storage import Pending,read,write
        c,folder,request,clip=self.make_prepared()
        bad={**request,'seed':17}
        write(folder/'request.json',bad)
        with self.assertRaises(Pending):
            flow.reconcile_prepared_direction(self.agent,clip,c)
        self.assertEqual(read(folder/'request.json'),bad)
        self.agent.s.save.assert_not_called()

    def test_late_replan_cannot_reintroduce_a_script_paraphrase_through_source_beats(self):
        from lite import dialogue_integrity, task_parameters
        from lite.media import video_request
        self.agent.config['completion_policy']['dialogue_integrity'] = dialogue_integrity.VERSION
        self.agent.config['task_contract'] = {'version':task_parameters.VERSION,'language':'English'}
        self.agent.config['generation_contract']['cast']['Yan']['voice_asset_id']='voice'
        line = {'speaker':'Yan','text':'Two hundred pounds.','on_screen_or_offscreen':'off_screen'}
        self.clip['dialogue']=[line]
        self.clip['render_spec']=spec('He says two hundred pounds. His wrist trembles.')
        self.clip['exit_state']='He says two hundred pounds. The paper falls.'
        self.record(3,render_spec=self.clip['render_spec'],exit_state=self.clip['exit_state'])
        view=flow.execution_clip(self.agent,self.clip)
        self.assertEqual(view,flow.execution_clip(self.agent,view))
        self.assertEqual(view['action_beats'][0]['action'],'His wrist trembles.')
        plan=flow.compile_plan(self.agent,view,{'render_spec':spec(self.old)})
        ref={'id':'voice','kind':'audio','sha256':'a'*64,'speaker':'Yan','track_type':'isolated_voice','path':'/voice'}
        # Exercise a cached/adversarial plan also at the final binding boundary.
        caption,_=binding_contract.compile(view,[ref],self.agent.config,{**plan,'render_spec':self.clip['render_spec']})
        prompt=video_request(caption,9,[ref],11101)['prompt']
        self.assertEqual(prompt.lower().count('two hundred pounds'),1)
        self.assertIn('<d>[English] Two hundred pounds.</d>',prompt)
        self.assertIn('Beat 1: His wrist trembles.',prompt)
        self.assertIn('exit_state target: The paper falls.',prompt)
