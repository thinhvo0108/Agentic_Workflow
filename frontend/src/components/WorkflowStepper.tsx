import {
  Box,
  Step,
  StepDescription,
  StepIcon,
  StepIndicator,
  StepNumber,
  StepSeparator,
  StepStatus,
  StepTitle,
  Stepper,
  Text,
} from '@chakra-ui/react';
import { keyframes } from '@emotion/react';
import type { WorkflowStatus } from '../types/workflow';

const pulse = keyframes`
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
`;

interface Step {
  nodes: string[];
  label: string;
  description: string;
}

const STEPS: Step[] = [
  { nodes: ['router'],              label: 'Routing',      description: 'Classifying query intent' },
  { nodes: ['research', 'support'], label: 'Analyzing',    description: 'Specialized agent processing' },
  { nodes: ['retriever'],           label: 'Retrieving',   description: 'Fetching relevant documents' },
  { nodes: ['reranker'],            label: 'Reranking',    description: 'Scoring document relevance' },
  { nodes: ['generator'],           label: 'Generating',   description: 'Drafting response' },
  { nodes: ['structured_output'],   label: 'Structuring',  description: 'Formatting output' },
  { nodes: ['groundedness'],        label: 'Groundedness', description: 'Verifying answer claims' },
  { nodes: ['checkpoint'],          label: 'Checkpoint',   description: 'Saving progress' },
  { nodes: ['auto_approval_gate'],  label: 'Gate',         description: 'Checking confidence threshold' },
  { nodes: ['human_approval'],      label: 'Review',       description: 'Awaiting approval' },
  { nodes: ['final_response'],      label: 'Complete',     description: 'Response finalized' },
  { nodes: ['knowledge_update'],    label: 'KB Update',    description: 'Adding to knowledge base' },
];

function getActiveIndex(currentNode: string | null, status: WorkflowStatus): number {
  if (status === 'completed') return STEPS.length;
  if (status === 'rejected')  return STEPS.length - 1;
  if (!currentNode) return 0;
  const idx = STEPS.findIndex((s) => s.nodes.includes(currentNode));
  return idx === -1 ? 0 : idx;
}

function resolveDescription(step: Step, status: WorkflowStatus, autoApproved?: boolean): string {
  const isGate   = step.nodes.includes('auto_approval_gate');
  const isReview = step.nodes.includes('human_approval');
  const isKB     = step.nodes.includes('knowledge_update');

  if (status === 'awaiting_approval') {
    if (isGate)   return 'Confidence below threshold';
    if (isReview) return 'Manual review required';
  }

  if ((status === 'completed' || status === 'rejected') && autoApproved !== undefined) {
    if (isGate)   return autoApproved ? 'Confidence ≥ 70%'          : 'Confidence below threshold';
    if (isReview) return autoApproved ? 'Auto-approved'              : 'Approved by reviewer';
    if (isKB)     return autoApproved ? 'Skipped — high confidence'  : 'Added to knowledge base';
  }

  return step.description;
}

interface Props {
  currentNode: string | null;
  status: WorkflowStatus;
  autoApproved?: boolean;
}

export default function WorkflowStepper({ currentNode, status, autoApproved }: Props) {
  const activeIndex = getActiveIndex(currentNode, status);
  const isRunning = status === 'running';

  return (
    <Box>
      <Text fontSize="xs" fontWeight="bold" textTransform="uppercase" color="gray.400" mb={4} letterSpacing="wider">
        Workflow Progress
      </Text>
      <Stepper index={activeIndex} orientation="vertical" colorScheme="brand" gap="0" size="sm">
        {STEPS.map((step, idx) => {
          const isActive   = idx === activeIndex && isRunning;
          const isGate     = step.nodes.includes('auto_approval_gate');
          const isReview   = step.nodes.includes('human_approval');
          const isKB       = step.nodes.includes('knowledge_update');
          const isPaused   = status === 'awaiting_approval';
          const isResolved = (status === 'completed' || status === 'rejected') && autoApproved !== undefined;
          const descriptionText = resolveDescription(step, status, autoApproved);

          return (
            <Step key={step.label}>
              <StepIndicator>
                <StepStatus
                  complete={<StepIcon />}
                  incomplete={<StepNumber />}
                  active={
                    <Box
                      as="span"
                      animation={isRunning ? `${pulse} 1.4s ease-in-out infinite` : undefined}
                    >
                      <StepNumber />
                    </Box>
                  }
                />
              </StepIndicator>

              <Box pb={6} minH="48px">
                <StepTitle>
                  <Text
                    fontSize="sm"
                    fontWeight={isActive || (isPaused && isReview) ? 'bold' : 'medium'}
                    color={
                      isActive                                ? 'brand.600'  :
                      isPaused && isGate                      ? 'orange.500' :
                      isPaused && isReview                    ? 'orange.600' :
                      isResolved && isReview && autoApproved  ? 'purple.600' :
                      isResolved && isReview && !autoApproved ? 'green.600'  :
                      isResolved && isKB     && !autoApproved ? 'green.600'  :
                      undefined
                    }
                  >
                    {step.label}
                  </Text>
                </StepTitle>
                <StepDescription>
                  <Text
                    fontSize="xs"
                    color={
                      isPaused   && (isGate || isReview)     ? 'orange.500' :
                      isResolved && isReview && autoApproved  ? 'purple.500' :
                      isResolved && isReview && !autoApproved ? 'green.600'  :
                      isResolved && isGate   && autoApproved  ? 'purple.500' :
                      isResolved && isKB     && !autoApproved ? 'green.600'  :
                      'gray.500'
                    }
                    fontWeight={isPaused && isReview ? 'medium' : 'normal'}
                  >
                    {descriptionText}
                  </Text>
                </StepDescription>
              </Box>

              <StepSeparator />
            </Step>
          );
        })}
      </Stepper>
    </Box>
  );
}
