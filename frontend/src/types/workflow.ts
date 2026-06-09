export type WorkflowStatus =
  | 'running'
  | 'awaiting_approval'
  | 'completed'
  | 'rejected'
  | 'failed'
  | 'not_found';

export type ApprovalAction = 'approved' | 'rejected';
export type RouteDecision = 'research' | 'support';

export interface Citation {
  document_id: string;
  source: string;
  excerpt: string;
  relevance_score: number;
}

export interface WorkflowStatusResponse {
  session_id: string;
  status: WorkflowStatus;
  current_node: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface WorkflowResponse {
  session_id: string;
  summary: string;
  answer: string;
  citations: Citation[];
  route: RouteDecision;
  approval_status: ApprovalAction;
  created_at: string;
}

export interface ApprovalResponse {
  session_id: string;
  action: ApprovalAction;
  reviewer_id: string;
  comment: string | null;
  processed_at: string;
}
