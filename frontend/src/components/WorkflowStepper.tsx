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
  { nodes: ['router'],             label: 'Routing',      description: 'Classifying query intent' },
  { nodes: ['research', 'support'],label: 'Analyzing',    description: 'Specialized agent processing' },
  { nodes: ['retriever'],          label: 'Retrieving',   description: 'Fetching relevant documents' },
  { nodes: ['reranker'],           label: 'Reranking',    description: 'Scoring document relevance' },
  { nodes: ['generator'],          label: 'Generating',   description: 'Drafting response' },
  { nodes: ['structured_output'],  label: 'Structuring',   description: 'Formatting output' },
  { nodes: ['groundedness'],       label: 'Groundedness',  description: 'Verifying answer claims' },
  { nodes: ['checkpoint'],         label: 'Checkpoint',    description: 'Saving progress' },
  { nodes: ['human_approval'],     label: 'Review',       description: 'Awaiting human approval' },
  { nodes: ['final_response'],     label: 'Complete',     description: 'Response finalized' },
];

function getActiveIndex(currentNode: string | null, status: WorkflowStatus): number {
  if (status === 'completed') return STEPS.length;
  if (status === 'rejected')  return STEPS.length - 1;
  if (!currentNode) return 0;
  const idx = STEPS.findIndex((s) => s.nodes.includes(currentNode));
  return idx === -1 ? 0 : idx;
}

interface Props {
  currentNode: string | null;
  status: WorkflowStatus;
}

export default function WorkflowStepper({ currentNode, status }: Props) {
  const activeIndex = getActiveIndex(currentNode, status);
  const isRunning = status === 'running';

  return (
    <Box>
      <Text fontSize="xs" fontWeight="bold" textTransform="uppercase" color="gray.400" mb={4} letterSpacing="wider">
        Workflow Progress
      </Text>
      <Stepper index={activeIndex} orientation="vertical" colorScheme="brand" gap="0" size="sm">
        {STEPS.map((step, idx) => {
          const isActive = idx === activeIndex && isRunning;
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
                    fontWeight={isActive ? 'bold' : 'medium'}
                    color={isActive ? 'brand.600' : undefined}
                  >
                    {step.label}
                  </Text>
                </StepTitle>
                <StepDescription>
                  <Text fontSize="xs" color="gray.500">
                    {step.description}
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
