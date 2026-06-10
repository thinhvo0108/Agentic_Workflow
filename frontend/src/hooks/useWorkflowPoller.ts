import { useCallback, useEffect, useRef, useState } from 'react';
import { getWorkflowStatus } from '../api/workflow';
import type { WorkflowStatus, WorkflowStatusResponse } from '../types/workflow';

const POLL_INTERVAL_MS = 1500;

// How many consecutive "not_found" responses to tolerate before giving up.
// AsyncPostgresSaver writes the initial checkpoint asynchronously, so the
// first poll(s) may arrive before any checkpoint exists.
const NOT_FOUND_PATIENCE = 6; // ~9 seconds

const STOP_POLLING: ReadonlySet<WorkflowStatus> = new Set([
  'completed',
  'rejected',
  'failed',
  'awaiting_approval',
]);

export function useWorkflowPoller(sessionId: string) {
  const [status, setStatus] = useState<WorkflowStatusResponse | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [pollKey, setPollKey] = useState(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const notFoundCountRef = useRef(0);

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
    notFoundCountRef.current = 0;
  }, [sessionId]);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const data = await getWorkflowStatus(sessionId);
        if (cancelled) return;

        if (data.status === 'not_found') {
          // The workflow task may have been created but AsyncPostgresSaver
          // hasn't committed the first checkpoint yet.  Keep polling silently
          // until the session appears or patience runs out.
          notFoundCountRef.current += 1;
          if (notFoundCountRef.current < NOT_FOUND_PATIENCE) {
            timeoutRef.current = setTimeout(poll, POLL_INTERVAL_MS);
            return;
          }
          // Patience exhausted — surface the not_found state.
        } else {
          notFoundCountRef.current = 0;
        }

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
