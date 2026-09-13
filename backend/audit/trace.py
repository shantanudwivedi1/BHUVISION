from datetime import datetime, timezone

from backend.models.schemas import AuditEntry, QueryPlan


def create_audit_entry(
    analysis_id: str,
    query: str,
    plan: QueryPlan,
    routing: dict,
    execution_status: str,
    errors: list[str],
    warnings: list[str],
) -> AuditEntry:
    return AuditEntry(
        analysis_id=analysis_id,
        query=query,
        plan=plan,
        routing_decision=routing,
        execution_status=execution_status,
        timestamp=datetime.now(timezone.utc),
        errors=errors,
        warnings=warnings,
    )
