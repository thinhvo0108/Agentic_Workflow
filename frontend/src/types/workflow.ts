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

export interface ConfidenceScores {
  router: number;
  retrieval: number;
  answer: number;
  overall: number;
}

export interface EvaluatedClaim {
  claim: string;
  supported: boolean;
  source_document_ids: string[];
  reasoning: string;
}

export interface GroundednessResult {
  groundedness_score: number;
  supported_claims: EvaluatedClaim[];
  unsupported_claims: EvaluatedClaim[];
  evaluated_at: string;
}

export interface WebSearchResult {
  title: string;
  link: string;
  snippet: string;
}

export interface DraftResponse {
  session_id: string;
  query: string;
  route: RouteDecision;
  summary: string;
  answer: string;
  citations: Citation[];
  confidence: ConfidenceScores | null;
  groundedness: GroundednessResult | null;
  judge_result: JudgeResult | null;
  web_search_results: WebSearchResult[];
}

export interface JudgeDimensionScore {
  score: number;
  reasoning: string;
}

export interface JudgeResult {
  faithfulness: JudgeDimensionScore;
  relevance: JudgeDimensionScore;
  completeness: JudgeDimensionScore;
  coherence: JudgeDimensionScore;
  overall_score: number;
  recommendation: 'auto_approve' | 'needs_review';
  critique: string;
  evaluated_at: string;
}

export interface WorkflowMetrics {
  started_at: string;
  completed_at: string;
  latency_ms: number;
  total_tokens: number;
  error_count: number;
  error_rate: number;
  hallucination_rate: number | null;
  judge_score: number | null;
  step_count: number;
}

export interface WorkflowResponse {
  session_id: string;
  summary: string;
  answer: string;
  citations: Citation[];
  route: RouteDecision;
  approval_status: ApprovalAction;
  auto_approved: boolean;
  knowledge_updated: boolean;
  reviewer_id: string | null;
  reviewer_comment: string | null;
  confidence: ConfidenceScores | null;
  groundedness: GroundednessResult | null;
  judge_result: JudgeResult | null;
  metrics: WorkflowMetrics | null;
  created_at: string;
}

export interface ApprovalResponse {
  session_id: string;
  action: ApprovalAction;
  reviewer_id: string;
  comment: string | null;
  processed_at: string;
}
