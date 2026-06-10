import { apiFetch } from './client';
import type {
  ApprovalAction,
  ApprovalResponse,
  DraftResponse,
  WorkflowResponse,
  WorkflowStatusResponse,
} from '../types/workflow';

export async function submitWorkflow(query: string): Promise<WorkflowStatusResponse> {
  return apiFetch<WorkflowStatusResponse>('/workflow', {
    method: 'POST',
    body: JSON.stringify({ query }),
  });
}

export async function getWorkflowStatus(sessionId: string): Promise<WorkflowStatusResponse> {
  return apiFetch<WorkflowStatusResponse>(`/workflow/${sessionId}`);
}

export async function getWorkflowResult(sessionId: string): Promise<WorkflowResponse> {
  return apiFetch<WorkflowResponse>(`/workflow/${sessionId}/result`);
}

export async function getWorkflowDraft(sessionId: string): Promise<DraftResponse> {
  return apiFetch<DraftResponse>(`/workflow/${sessionId}/draft`);
}

export async function submitApproval(
  sessionId: string,
  action: ApprovalAction,
  reviewerId: string,
  comment?: string,
): Promise<ApprovalResponse> {
  return apiFetch<ApprovalResponse>(`/workflow/${sessionId}/approve`, {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      action,
      reviewer_id: reviewerId,
      comment: comment ?? null,
    }),
  });
}
