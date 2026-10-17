# Assessment orchestration

The orchestrator coordinates module use cases without moving their business
rules into a central service. Cube vision and behaviour analysis can execute
concurrently for the same task. Their derived results are joined by
`pseudonymous_child_id`, `session_id`, and `task_id` before a profile update.

Persisted internal events or framework background tasks should trigger durable,
long-running profile work; deployments that need stronger delivery guarantees
can replace the adapter without changing domain modules.
