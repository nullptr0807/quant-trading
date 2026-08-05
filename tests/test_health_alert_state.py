def test_health_alert_state_emits_new_escalation_and_recovery(tmp_path):
    from scripts.classify_health_alerts import process
    state=tmp_path/'state.json'
    issue={'severity':'critical','check':'x','market':'US'}
    first=process({'issues':[issue]},state)
    assert first['notify'][0]['transition']=='new'
    assert process({'issues':[issue]},state)['notify']==[]
    third=process({'issues':[issue]},state)
    assert third['notify'][0]['transition']=='continuing'
    recovered=process({'issues':[]},state)
    assert recovered['notify'][0]['transition']=='recovered'
    assert state.stat().st_mode & 0o777==0o600
