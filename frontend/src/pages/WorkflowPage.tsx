import { useEffect, useState } from 'react';
import { Link as RouterLink, useParams } from 'react-router-dom';
import {
  Alert,
  AlertDescription,
  AlertIcon,
  AlertTitle,
  Badge,
  Box,
  Button,
  Card,
  CardBody,
  Container,
  Flex,
  Grid,
  HStack,
  Icon,
  Skeleton,
  Spinner,
  Text,
  VStack,
} from '@chakra-ui/react';
import { ArrowBackIcon, SearchIcon, WarningTwoIcon } from '@chakra-ui/icons';
import { keyframes } from '@emotion/react';
import WorkflowStepper from '../components/WorkflowStepper';
import ApprovalPanel from '../components/ApprovalPanel';
import FinalResponsePanel from '../components/FinalResponsePanel';
import { useWorkflowPoller } from '../hooks/useWorkflowPoller';
import { getWorkflowResult, submitApproval } from '../api/workflow';
import type { ApprovalAction, WorkflowResponse } from '../types/workflow';

const shimmer = keyframes`
  0%   { background-position: -200% 0; }
  100% { background-position:  200% 0; }
`;

const STATUS_META: Record<string, { label: string; color: string }> = {
  running:           { label: 'Running',          color: 'blue' },
  awaiting_approval: { label: 'Awaiting Approval', color: 'orange' },
  completed:         { label: 'Completed',         color: 'green' },
  rejected:          { label: 'Rejected',          color: 'red' },
  failed:            { label: 'Failed',            color: 'red' },
  not_found:         { label: 'Not Found',         color: 'gray' },
};

export default function WorkflowPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const { status, fetchError, refetch } = useWorkflowPoller(sessionId ?? '');
  const [result, setResult] = useState<WorkflowResponse | null>(null);
  const [resultError, setResultError] = useState<string | null>(null);

  // Fetch the final result once the workflow completes
  useEffect(() => {
    if (status?.status !== 'completed' || !sessionId) return;
    getWorkflowResult(sessionId)
      .then(setResult)
      .catch((err: unknown) =>
        setResultError(err instanceof Error ? err.message : 'Failed to load result.'),
      );
  }, [sessionId, status?.status]);

  const handleApproval = async (
    action: ApprovalAction,
    reviewerId: string,
    comment?: string,
  ) => {
    await submitApproval(sessionId ?? '', action, reviewerId, comment);
    refetch();
  };

  if (!sessionId) {
    return (
      <Container maxW="xl" py={16} textAlign="center">
        <Text color="red.500">Invalid session URL.</Text>
        <Button as={RouterLink} to="/" mt={4} variant="ghost" leftIcon={<ArrowBackIcon />}>
          Back to Home
        </Button>
      </Container>
    );
  }

  return (
    <Flex direction="column" minH="100vh">
      {/* Header */}
      <Box bg="brand.600" py={4} px={6} shadow="md">
        <HStack justify="space-between">
          <HStack spacing={3}>
            <Icon as={SearchIcon} color="white" boxSize={5} />
            <Text fontWeight="bold" fontSize="lg" color="white" letterSpacing="tight">
              Agentic Workflow
            </Text>
          </HStack>
          <Button
            as={RouterLink}
            to="/"
            size="sm"
            variant="ghost"
            color="brand.100"
            _hover={{ bg: 'brand.700', color: 'white' }}
            leftIcon={<ArrowBackIcon />}
          >
            New query
          </Button>
        </HStack>
      </Box>

      <Container maxW="6xl" px={{ base: 4, md: 6 }} py={8} flex="1">
        {/* Session info strip */}
        <HStack justify="space-between" mb={6} wrap="wrap" gap={2}>
          <VStack align="start" spacing={0}>
            <Text fontSize="xs" color="gray.400" textTransform="uppercase" letterSpacing="wider">
              Session
            </Text>
            <Text fontSize="sm" fontFamily="mono" color="gray.600" noOfLines={1}>
              {sessionId}
            </Text>
          </VStack>
          {status && (
            <Badge
              colorScheme={STATUS_META[status.status]?.color ?? 'gray'}
              fontSize="sm"
              px={3}
              py={1}
              borderRadius="full"
              variant="subtle"
            >
              {STATUS_META[status.status]?.label ?? status.status}
            </Badge>
          )}
        </HStack>

        {/* Network error */}
        {fetchError && (
          <Alert status="error" borderRadius="lg" mb={6}>
            <AlertIcon />
            <AlertTitle>Connection error</AlertTitle>
            <AlertDescription fontSize="sm">{fetchError}</AlertDescription>
          </Alert>
        )}

        {/* Main 2-col layout */}
        <Grid templateColumns={{ base: '1fr', lg: '260px 1fr' }} gap={8}>
          {/* Left: Stepper */}
          <Box
            bg="white"
            borderRadius="xl"
            border="1px solid"
            borderColor="gray.200"
            p={6}
            alignSelf="start"
            position={{ lg: 'sticky' }}
            top="24px"
          >
            {status ? (
              <WorkflowStepper
                currentNode={status.current_node}
                status={status.status}
              />
            ) : (
              <VStack spacing={4} align="stretch">
                {Array.from({ length: 5 }).map((_, i) => (
                  <HStack key={i} spacing={3}>
                    <Skeleton borderRadius="full" boxSize={6} />
                    <VStack align="stretch" flex={1} spacing={1}>
                      <Skeleton h={3} w="60%" />
                      <Skeleton h={2} w="80%" />
                    </VStack>
                  </HStack>
                ))}
              </VStack>
            )}
          </Box>

          {/* Right: Content */}
          <Box>
            <Card borderRadius="xl" shadow="sm" overflow="hidden">
              <CardBody p={7}>
                {!status && !fetchError && <RunningView node={null} />}

                {status?.status === 'running' && (
                  <RunningView node={status.current_node} />
                )}

                {status?.status === 'awaiting_approval' && (
                  <ApprovalPanel
                    sessionId={sessionId}
                    query=""
                    onDecision={handleApproval}
                  />
                )}

                {status?.status === 'completed' && result && (
                  <FinalResponsePanel result={result} />
                )}

                {status?.status === 'completed' && !result && !resultError && (
                  <RunningView node="Loading result…" />
                )}

                {status?.status === 'completed' && resultError && (
                  <ErrorView title="Failed to load result" message={resultError} />
                )}

                {status?.status === 'rejected' && (
                  <RejectedView sessionId={sessionId} />
                )}

                {status?.status === 'failed' && (
                  <ErrorView
                    title="Workflow failed"
                    message={status.error ?? 'An unexpected error occurred.'}
                  />
                )}

                {status?.status === 'not_found' && (
                  <ErrorView
                    title="Session not found"
                    message={`No workflow session exists for ID: ${sessionId}`}
                  />
                )}
              </CardBody>
            </Card>
          </Box>
        </Grid>
      </Container>
    </Flex>
  );
}

// ── Sub-views ──────────────────────────────────────────────────────────────────

function RunningView({ node }: { node: string | null }) {
  const shimmerBg =
    'linear-gradient(90deg, transparent 0%, rgba(124,58,237,0.06) 50%, transparent 100%)';

  return (
    <VStack spacing={8} py={8} align="center" textAlign="center">
      <Box
        w={16}
        h={16}
        borderRadius="full"
        bg="brand.50"
        border="3px solid"
        borderColor="brand.200"
        display="flex"
        alignItems="center"
        justifyContent="center"
        backgroundImage={shimmerBg}
        backgroundSize="200% 100%"
        animation={`${shimmer} 1.8s linear infinite`}
      >
        <Spinner color="brand.500" size="lg" thickness="3px" />
      </Box>

      <VStack spacing={2}>
        <Text fontWeight="bold" fontSize="xl" color="gray.700">
          Processing your query
        </Text>
        <Text fontSize="sm" color="gray.500">
          The multi-agent workflow is running. This usually takes 15–60 seconds.
        </Text>
      </VStack>

      {node && (
        <Box bg="gray.50" borderRadius="lg" px={5} py={3}>
          <NodeLabel node={node} />
        </Box>
      )}
    </VStack>
  );
}

function RejectedView({ sessionId }: { sessionId: string }) {
  return (
    <VStack spacing={5} py={8} align="center" textAlign="center">
      <Text fontSize="4xl">🚫</Text>
      <VStack spacing={2}>
        <Text fontWeight="bold" fontSize="xl" color="red.600">
          Response Rejected
        </Text>
        <Text fontSize="sm" color="gray.500">
          The reviewer has rejected the AI-generated response for session{' '}
          <Text as="span" fontFamily="mono" fontSize="xs">
            {sessionId}
          </Text>
          .
        </Text>
      </VStack>
      <Button as={RouterLink} to="/" colorScheme="brand" leftIcon={<ArrowBackIcon />}>
        Start a new query
      </Button>
    </VStack>
  );
}

function ErrorView({ title, message }: { title: string; message: string }) {
  return (
    <VStack spacing={5} py={8} align="center" textAlign="center">
      <WarningTwoIcon color="red.400" boxSize={12} />
      <VStack spacing={2}>
        <Text fontWeight="bold" fontSize="xl" color="red.600">
          {title}
        </Text>
        <Text fontSize="sm" color="gray.500" maxW="md">
          {message}
        </Text>
      </VStack>
      <Button as={RouterLink} to="/" colorScheme="brand" leftIcon={<ArrowBackIcon />}>
        Try again
      </Button>
    </VStack>
  );
}

function NodeLabel({ node }: { node: string }) {
  const pretty = node.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  return (
    <HStack spacing={2}>
      <Spinner size="xs" color="brand.500" />
      <Text fontSize="sm" color="gray.600">
        Processing:{' '}
        <Text as="span" fontWeight="semibold" color="brand.600">
          {pretty}
        </Text>
      </Text>
    </HStack>
  );
}
