import { useCallback, useEffect, useRef, useState } from 'react';
import { getWorkflowStatus } from '../api/workflow';
import type { WorkflowStatus, WorkflowStatusResponse } from '../types/workflow';

const POLL_INTERVAL_MS = 1500;

const STOP_POLLING: ReadonlySet<WorkflowStatus> = new Set([
  'completed',
  'rejected',
  'failed',
  'not_found',
  'awaiting_approval',
]);

export function useWorkflowPoller(sessionId: string) {
  const [status, setStatus] = useState<WorkflowStatusResponse | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [pollKey, setPollKey] = useState(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearScheduled = () => {
    if (timeoutRef.current !== null) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  };

  // Reset status when sessionId changes so a stale "completed" from a previous
  // session never triggers an immediate /result fetch on a new session.
  useEffect(() => {
    setStatus(null);
    setFetchError(null);
  }, [sessionId]);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const data = await getWorkflowStatus(sessionId);
        if (cancelled) return;
        setStatus(data);
        setFetchError(null);
        if (!STOP_POLLING.has(data.status)) {
          timeoutRef.current = setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch (err) {
        if (cancelled) return;
        setFetchError(err instanceof Error ? err.message : 'Unknown error');
      }
    }

    poll();

    return () => {
      cancelled = true;
      clearScheduled();
    };
  }, [sessionId, pollKey]);

  const refetch = useCallback(() => {
    clearScheduled();
    setPollKey((k) => k + 1);
  }, []);

  return { status, fetchError, refetch };
}
