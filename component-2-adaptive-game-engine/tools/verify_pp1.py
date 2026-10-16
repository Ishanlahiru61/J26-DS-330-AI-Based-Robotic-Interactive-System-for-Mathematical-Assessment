"""Run actual tests and save deterministic local workflow examples for handover."""
import io
import json
import platform
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
OUTPUT=ROOT/'docs/integration_evidence'
OUTPUT.mkdir(parents=True,exist_ok=True)
from component2_app.service import Component2Service


def main():
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (OUTPUT/'test_results.txt').write_text(stream.getvalue(),encoding='utf-8')
    report={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'skipped':len(result.skipped),'success':result.wasSuccessful(),'python':platform.python_version()}
    if result.wasSuccessful():
        with tempfile.TemporaryDirectory() as temp:
            service=Component2Service(Path(temp)/'demo.db')
            demos=[]
            for grade in (1,2,3,4,5):
                lid=f'PP1_GRADE_{grade}'
                service.db.upsert_learner(lid,lid,grade,True)
                sid=service.start_or_resume(lid)['session_id']
                trace=[]
                for index in range(36):
                    task=service.next_task(sid,index)
                    with service.db.connect() as db:
                        answer=db.execute('SELECT expected_answer FROM question_instances WHERE question_id=?',(task['question_id'],)).fetchone()[0]
                    response=service.submit(sid,task['question_id'],answer if index%5 else 0)
                    trace.append({'recommendation':task['recommendation'],'question':task['robot_task'],'response':response})
                profile=service.export_c4(lid,sid)
                (OUTPUT/f'grade_{grade}_recommendation.json').write_text(json.dumps(trace[-1]['recommendation'],indent=2),encoding='utf-8')
                (OUTPUT/f'grade_{grade}_c4_profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
                (OUTPUT/f'grade_{grade}_workflow.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
                demos.append({'grade':grade,'responses':36,'games_seen':sorted({r['recommendation']['game_type'] for r in trace}),
                              'areas':[{k:a[k] for k in ('area_id','attempts','coverage_status')} for a in profile['profile']['areas']]})
            report['demonstrations']=demos
    (OUTPUT/'verification_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not result.wasSuccessful():
        print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__': raise SystemExit(main())
