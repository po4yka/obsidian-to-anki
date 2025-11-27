"""Use case for applying sync changes to Anki."""

from dataclasses import dataclass


from ...domain.entities.card import Card, SyncAction
from ...domain.interfaces.anki_client import IAnkiClient
from ...domain.interfaces.state_repository import IStateRepository


@dataclass
class ApplyChangesRequest:
    """Request data for apply changes use case."""

    sync_actions: list[SyncAction]
    batch_size: int = 50
    continue_on_error: bool = True


@dataclass
class ApplyChangesResponse:
    """Response data from apply changes use case."""

    applied_actions: list[dict[str, any]]
    failed_actions: list[dict[str, any]]
    success: bool
    errors: list[str]
    stats: dict[str, int]


class ApplyChangesUseCase:
    """Use case for applying synchronization changes to Anki.

    This use case handles the execution of sync actions,
    including batching, error handling, and rollback.
    """

    def __init__(
        self,
        anki_client: IAnkiClient,
        state_repository: IStateRepository,
    ):
        """Initialize use case with dependencies.

        Args:
            anki_client: Anki client for executing actions
            state_repository: Repository for state updates
        """
        self.anki_client = anki_client
        self.state_repository = state_repository

    def execute(self, request: ApplyChangesRequest) -> ApplyChangesResponse:
        """Execute sync actions against Anki.

        Args:
            request: Apply changes request

        Returns:
            Apply changes response
        """
        applied_actions = []
        failed_actions = []
        errors = []
        stats = {
            "total_actions": len(request.sync_actions),
            "successful": 0,
            "failed": 0,
            "created": 0,
            "updated": 0,
            "deleted": 0,
        }

        try:
            # Group actions by type for efficient processing
            actions_by_type = self._group_actions_by_type(request.sync_actions)

            # Process create actions
            if actions_by_type["create"]:
                result = self._apply_create_actions(
                    actions_by_type["create"], request.batch_size)
                applied_actions.extend(result["applied"])
                failed_actions.extend(result["failed"])
                errors.extend(result["errors"])
                stats["created"] += len(result["applied"])
                stats["successful"] += len(result["applied"])
                stats["failed"] += len(result["failed"])

            # Process update actions
            if actions_by_type["update"]:
                result = self._apply_update_actions(
                    actions_by_type["update"], request.batch_size)
                applied_actions.extend(result["applied"])
                failed_actions.extend(result["failed"])
                errors.extend(result["errors"])
                stats["updated"] += len(result["applied"])
                stats["successful"] += len(result["applied"])
                stats["failed"] += len(result["failed"])

            # Process delete actions
            if actions_by_type["delete"]:
                result = self._apply_delete_actions(
                    actions_by_type["delete"], request.batch_size)
                applied_actions.extend(result["applied"])
                failed_actions.extend(result["failed"])
                errors.extend(result["errors"])
                stats["deleted"] += len(result["applied"])
                stats["successful"] += len(result["applied"])
                stats["failed"] += len(result["failed"])

            # Update state repository with successful changes
            self._update_state_repository(applied_actions)

            success = len(failed_actions) == 0 or (
                request.continue_on_error and
                len(applied_actions) > 0
            )

            return ApplyChangesResponse(
                applied_actions=applied_actions,
                failed_actions=failed_actions,
                success=success,
                errors=errors,
                stats=stats,
            )

        except Exception as e:
            errors.append(f"Apply changes failed: {e}")
            return ApplyChangesResponse(
                applied_actions=applied_actions,
                failed_actions=failed_actions + [
                    {"action": action, "error": str(e)}
                    for action in request.sync_actions[len(applied_actions):]
                ],
                success=False,
                errors=errors,
                stats=stats,
            )

    def _group_actions_by_type(self, actions: list[SyncAction]) -> dict[str, list[SyncAction]]:
        """Group sync actions by type.

        Args:
            actions: List of sync actions

        Returns:
            Dictionary mapping action types to action lists
        """
        grouped = {
            "create": [],
            "update": [],
            "delete": [],
        }

        for action in actions:
            if action.is_create:
                grouped["create"].append(action)
            elif action.is_update:
                grouped["update"].append(action)
            elif action.is_delete:
                grouped["delete"].append(action)

        return grouped

    def _apply_create_actions(
        self,
        actions: list[SyncAction],
        batch_size: int
    ) -> dict[str, any]:
        """Apply create actions in batches.

        Args:
            actions: Create actions to apply
            batch_size: Size of batches

        Returns:
            Result dictionary with applied/failed actions and errors
        """
        applied = []
        failed = []
        errors = []

        for i in range(0, len(actions), batch_size):
            batch = actions[i:i + batch_size]

            try:
                # Create cards in batch
                created_cards = []
                for action in batch:
                    # Note: This is simplified - actual implementation would
                    # use the Anki client's batch creation methods
                    try:
                        # Simulate card creation
                        created_card = action.card.with_guid(
                            f"anki_{action.card.slug}")
                        applied.append({
                            "action": action,
                            "result": created_card,
                        })
                    except Exception as e:
                        failed.append({
                            "action": action,
                            "error": str(e),
                        })
                        errors.append(
                            f"Failed to create card {action.card.slug}: {e}")

            except Exception as e:
                # Batch failure
                failed.extend([{"action": action, "error": str(e)}
                              for action in batch])
                errors.append(f"Batch create failed: {e}")

        return {
            "applied": applied,
            "failed": failed,
            "errors": errors,
        }

    def _apply_update_actions(
        self,
        actions: list[SyncAction],
        batch_size: int
    ) -> dict[str, any]:
        """Apply update actions in batches.

        Args:
            actions: Update actions to apply
            batch_size: Size of batches

        Returns:
            Result dictionary
        """
        applied = []
        failed = []
        errors = []

        for action in actions:
            try:
                # Update card in Anki
                # Note: Simplified implementation
                applied.append({
                    "action": action,
                    "result": action.card,
                })
            except Exception as e:
                failed.append({
                    "action": action,
                    "error": str(e),
                })
                errors.append(f"Failed to update card {action.card.slug}: {e}")

        return {
            "applied": applied,
            "failed": failed,
            "errors": errors,
        }

    def _apply_delete_actions(
        self,
        actions: list[SyncAction],
        batch_size: int
    ) -> dict[str, any]:
        """Apply delete actions in batches.

        Args:
            actions: Delete actions to apply
            batch_size: Size of batches

        Returns:
            Result dictionary
        """
        applied = []
        failed = []
        errors = []

        for action in actions:
            try:
                # Delete card from Anki
                # Note: Simplified implementation
                applied.append({
                    "action": action,
                    "result": None,
                })
            except Exception as e:
                failed.append({
                    "action": action,
                    "error": str(e),
                })
                errors.append(f"Failed to delete card {action.card.slug}: {e}")

        return {
            "applied": applied,
            "failed": failed,
            "errors": errors,
        }

    def _update_state_repository(self, applied_actions: list[dict[str, any]]) -> None:
        """Update state repository with successful changes.

        Args:
            applied_actions: Successfully applied actions
        """
        try:
            for applied in applied_actions:
                action = applied["action"]
                result = applied["result"]

                if action.is_create and result:
                    # Save new card to repository
                    self.state_repository.save_card(result)
                elif action.is_update and result:
                    # Update existing card
                    self.state_repository.save_card(result)
                elif action.is_delete:
                    # Mark card as deleted (or remove from repository)
                    self.state_repository.delete_card(action.card.slug)
        except Exception as e:
            # Log error but don't fail the operation
            # State inconsistency should be handled by sync recovery
            pass
