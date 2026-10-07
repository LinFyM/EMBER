"""Scene pairing tolerates uint8 quantization but rejects physical/image changes."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from ember.pi05_eval import scene

@pytest.mark.parametrize('failure', ['stale_rgb', 'real_rgb', 'physics'])
def test_real_restore_entry_refreshes_only_rgb_and_preserves_strict_pairing(tmp_path,monkeypatch,failure):
    path=scene.scene_path(tmp_path,{'suite':'libero_spatial','task_id':6},24)
    np.savez(path, initial_rgb_canonical180=np.zeros((2,256,256,3),dtype=np.uint8))
    calls=[];physics=object();stale={'rgb':'stale','agentview_image':np.ones((256,256,3),dtype=np.uint8),'robot0_eye_in_hand_image':np.ones((256,256,3),dtype=np.uint8)};fresh=stale|{'rgb':'fresh'}
    owner=SimpleNamespace(obj_body_id={},parsed_problem={'goal_state':[]},sim=physics)
    def get_observations(*,force_update):
        assert force_update is True and owner.sim is physics
        calls.append('refresh');return fresh
    owner._get_observations=get_observations
    env=SimpleNamespace(env=owner)
    monkeypatch.setattr(scene,'_restore_scene',lambda *_:stale)
    def check(_env,ob,_names,_goals,snapshot,*,image):
        calls.append(('check',image,ob['rgb']))
        if failure=='physics':raise ValueError('restored full-scene start differs: initial_body_pos')
        if image and (ob is stale or failure=='real_rgb'):
            raise ValueError('restored full-scene start differs: initial_rgb_canonical180')
        if not image:assert 'initial_rgb_canonical180' not in snapshot
    monkeypatch.setattr(scene,'_assert_scene_pair',check)
    if failure=='stale_rgb':
        ob,_=scene.restore_registered_scene(env,stale,{'suite':'libero_spatial','task_id':6},24,tmp_path)
        assert ob is fresh and calls.count('refresh')==1
    else:
        with pytest.raises(ValueError):scene.restore_registered_scene(env,stale,{'suite':'libero_spatial','task_id':6},24,tmp_path,diagnostic_output=tmp_path/'failures')
        assert len(list((tmp_path/'failures').glob('*.npz'))) == (0 if failure=='physics' else 1)
        assert calls.count('refresh')==(0 if failure=='physics' else 1)


def test_registered_scene_accepts_one_color_level_without_relaxing_geometry(monkeypatch):
    expected = {"initial_rgb_canonical180": np.zeros((2, 256, 256, 3), dtype=np.uint8),
                "initial_eef_pos": np.zeros(3)}
    actual = {key: value.copy() for key, value in expected.items()}
    actual["initial_rgb_canonical180"][1, 5, 7] = 1
    monkeypatch.setattr(scene, "_scene_snapshot", lambda *args, **kwargs: actual)
    scene._assert_scene_pair(None, {}, [], [], expected, image=True)
    actual["initial_rgb_canonical180"][1, 5, 7] = 2
    with pytest.raises(ValueError, match="initial_rgb_canonical180"):
        scene._assert_scene_pair(None, {}, [], [], expected, image=True)
    actual["initial_rgb_canonical180"][1, 5, 7] = 1
    actual["initial_eef_pos"][0] = 2e-8
    with pytest.raises(ValueError, match="initial_eef_pos"):
        scene._assert_scene_pair(None, {}, [], [], expected, image=True)
