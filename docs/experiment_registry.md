# Persistent paper experiments

Approved rule parameters are stored in analysis/adaptive_restrictions/experiment_registry.json.
New recommendation evidence cannot remove or change an existing approved rule.
Explicit wait, reject and freeze decisions pause application, not outcome history.
Measured STOP_PAPER remains suspended. Promotion decisions use a separate source
and do not stop the underlying experiment.

Migration seeds currently active tests from the previous restriction report.
Historical approval snapshots recover vanished tests as recovered_pending_review;
they are measured and displayed, but are not silently reactivated. An explicit
new approval for the original source is required to resume them. The UI does not
yet offer a dedicated resume action for proposals no longer in the queue.

Live enforcement is unchanged. Registry read errors fail the build instead of
overwriting the registry. Historical event records must remain in the database.
