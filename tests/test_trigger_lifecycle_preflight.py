"""Actual trigger lifecycle must gate every public/helper write."""
import boto3
from botocore.stub import Stubber
import pytest
from cyint_aws_ml_ops_tools.glue import jobs

BAD=[("SCHEDULED", state) for state in ("CREATING","DELETING","UPDATING","ACTIVATING","DEACTIVATING","unknown",None,"missing")]
BAD += [("ON_DEMAND", "ACTIVATED"),("ON_DEMAND", "DEACTIVATED")]

def definition(kind):
    return {"Schedule": "cron(0 12 * * ? *)"} if kind == "SCHEDULED" else {}

@pytest.mark.parametrize("kind,state", BAD)
@pytest.mark.parametrize("entrypoint", ["public", "direct"])
def test_unconfirmed_trigger_lifecycle_stops_before_any_write(mocker,kind,state,entrypoint):
    glue,storage=mocker.Mock(),mocker.Mock()
    current={"Type":kind}
    if state!="missing":current["State"]=state
    glue.get_trigger.return_value={"Trigger":current}
    storage.meta.region_name="us-east-1"
    mocker.patch.object(jobs.boto3,"client",side_effect=[glue,storage])
    with pytest.raises(RuntimeError,match="stable"):
        if entrypoint=="public":
            jobs.define_job("job","glueetl","script.py","script.py","trigger",kind,definition(kind),
                aws_access_key="testing",aws_secret_key="testing",region="us-east-1",role="qualification",
                glue_script_bucket="scripts",environment_prefix="qualification-")
        else:jobs.handle_trigger(glue,"job","trigger",kind,definition(kind))
    assert storage.mock_calls==[]
    glue.update_job.assert_not_called()
    glue.create_job.assert_not_called()
    glue.create_trigger.assert_not_called()
    glue.update_trigger.assert_not_called()

@pytest.mark.parametrize("kind,state",[("ON_DEMAND","CREATED"),("SCHEDULED","CREATED"),("SCHEDULED","ACTIVATED"),("SCHEDULED","DEACTIVATED")])
def test_documented_stable_states_preserve_actual_sdk_update(kind,state):
    client=boto3.client("glue",region_name="us-east-1")
    update={**definition(kind),"Actions":[{"JobName":"job"}]}
    with Stubber(client) as stub:
        stub.add_response("get_trigger",{"Trigger":{"Name":"trigger","Type":kind,"State":state}},{"Name":"trigger"})
        stub.add_response("update_trigger",{},{"Name":"trigger","TriggerUpdate":update})
        jobs.handle_trigger(client,"job","trigger",kind,definition(kind))
        stub.assert_no_pending_responses()
