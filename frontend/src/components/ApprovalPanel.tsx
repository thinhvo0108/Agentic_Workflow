import { useEffect, useState } from 'react';
import {
  Alert,
  AlertIcon,
  Badge,
  Box,
  Button,
  Card,
  CardBody,
  Divider,
  FormControl,
  FormHelperText,
  FormLabel,
  HStack,
  Input,
  Skeleton,
  SkeletonText,
  Text,
  Textarea,
  VStack,
} from '@chakra-ui/react';
import { AttachmentIcon, CheckCircleIcon, CloseIcon, ExternalLinkIcon, SearchIcon, WarningTwoIcon } from '@chakra-ui/icons';
import { getWorkflowDraft } from '../api/workflow';
import type { ApprovalAction, DraftResponse } from '../types/workflow';
import ConfidenceStats, { scoreColor } from './ConfidenceStats';

const CONFIDENCE_THRESHOLD = 0.70;
const JUDGE_THRESHOLD = 0.60;

const ROUTE_LABELS: Record<string, { label: string; color: string }> = {
  research: { label: 'Research Agent', color: 'blue' },
  support:  { label: 'Support Agent',  color: 'teal' },
};

interface Props {
  sessionId: string;
  query: string;
  onDecision: (action: ApprovalAction, reviewerId: string, comment?: string, editedAnswer?: string) => Promise<void>;
}

export default function ApprovalPanel({ sessionId, query, onDecision }: Props) {
  const [draft, setDraft]               = useState<DraftResponse | null>(null);
  const [draftLoading, setDraftLoading] = useState(true);
  const [draftError, setDraftError]     = useState<string | null>(null);

  const [editedAnswer, setEditedAnswer]   = useState('');
  const [reviewerId, setReviewerId]       = useState('');
  const [comment, setComment]             = useState('');
  const [isLoading, setIsLoading]         = useState(false);
  const [pendingAction, setPendingAction] = useState<ApprovalAction | null>(null);
  const [formError, setFormError]         = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setDraftLoading(true);
    setDraftError(null);
    getWorkflowDraft(sessionId)
      .then((d) => {
        if (!cancelled) {
          setDraft(d);
          setEditedAnswer(d.answer);
          setDraftLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setDraftError(err instanceof Error ? err.message : 'Failed to load draft.');
          setDraftLoading(false);
        }
      });
    return () => { cancelled = true; };
  }, [sessionId]);

  const handleDecision = async (action: ApprovalAction) => {
    if (!reviewerId.trim()) { setFormError('Reviewer ID is required.'); return; }
    setFormError(null);
    setIsLoading(true);
    setPendingAction(action);
    try {
      await onDecision(
        action,
        reviewerId.trim(),
        comment.trim() || undefined,
        action === 'approved' ? editedAnswer.trim() || undefined : undefined,
      );
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Approval failed. Please try again.');
      setIsLoading(false);
      setPendingAction(null);
    }
  };

  const route        = draft ? (ROUTE_LABELS[draft.route] ?? { label: draft.route, color: 'gray' }) : null;
  const overallScore = draft?.confidence?.overall ?? null;
  const judgeScore   = draft?.judge_result?.overall_score ?? null;
  const confidenceFailed = overallScore !== null && overallScore < CONFIDENCE_THRESHOLD;
  const judgeFailed      = judgeScore   !== null && judgeScore   < JUDGE_THRESHOLD;

  return (
    <VStack align="stretch" spacing={5}>

      {/* ── Header ── */}
      <HStack justify="space-between" wrap="wrap" gap={2}>
        <VStack align="start" spacing={0}>
          <Text fontSize="xs" fontWeight="bold" textTransform="uppercase" color="orange.500" letterSpacing="wider">
            Awaiting Approval
          </Text>
          <Text fontSize="sm" color="gray.600">
            Review the AI-generated response before releasing it.
          </Text>
        </VStack>
        {route && (
          <Badge colorScheme={route.color} variant="subtle" fontSize="xs" px={2} py={1} borderRadius="full">
            {route.label}
          </Badge>
        )}
      </HStack>

      {/* ── Manual review reason ── */}
      <Box bg="orange.50" border="1px solid" borderColor="orange.200" borderRadius="lg" p={4}>
        <HStack spacing={3} align="start">
          <WarningTwoIcon color="orange.400" mt={0.5} flexShrink={0} />
          <VStack align="start" spacing={1}>
            <Text fontSize="xs" fontWeight="bold" color="orange.700">
              Manual review required
            </Text>
            {draft ? (
              <Text fontSize="xs" color="orange.600" lineHeight="tall">
                {confidenceFailed && judgeFailed ? (
                  <>
                    Confidence{' '}
                    <Text as="span" fontWeight="bold" color="orange.700">{Math.round(overallScore! * 100)}%</Text>
                    {' '}(threshold {Math.round(CONFIDENCE_THRESHOLD * 100)}%) and judge score{' '}
                    <Text as="span" fontWeight="bold" color="orange.700">{Math.round(judgeScore! * 100)}%</Text>
                    {' '}(threshold {Math.round(JUDGE_THRESHOLD * 100)}%) are both below their thresholds.
                  </>
                ) : judgeFailed ? (
                  <>
                    Confidence is sufficient{overallScore !== null ? <>{' '}(<Text as="span" fontWeight="bold" color="orange.700">{Math.round(overallScore * 100)}%</Text>)</> : null}
                    {' '}but the LLM judge score{' '}
                    <Text as="span" fontWeight="bold" color="orange.700">{Math.round(judgeScore! * 100)}%</Text>
                    {' '}is below the <Text as="span" fontWeight="bold">{Math.round(JUDGE_THRESHOLD * 100)}%</Text> quality threshold.
                  </>
                ) : confidenceFailed ? (
                  <>
                    Overall confidence{' '}
                    <Text as="span" fontWeight="bold" color="orange.700">{Math.round(overallScore! * 100)}%</Text>
                    {' '}is below the <Text as="span" fontWeight="bold">{Math.round(CONFIDENCE_THRESHOLD * 100)}%</Text> auto-approval threshold.
                  </>
                ) : (
                  <>One or more quality signals did not meet the auto-approval threshold.</>
                )}
                {' '}Please review carefully before deciding.
              </Text>
            ) : (
              <Text fontSize="xs" color="orange.600">
                Evaluating quality signals… Loading details.
              </Text>
            )}
          </VStack>
        </HStack>
      </Box>

      {/* ── Query ── */}
      <Box bg="purple.50" border="1px solid" borderColor="purple.200" borderRadius="lg" p={4}>
        <Text fontSize="2xs" fontWeight="bold" textTransform="uppercase" color="purple.400" letterSpacing="wider" mb={1}>
          Query
        </Text>
        <Text fontSize="sm" color="purple.800" fontStyle="italic">
          "{query || draft?.query}"
        </Text>
      </Box>

      {/* ── Draft skeleton ── */}
      {draftLoading && (
        <VStack align="stretch" spacing={3}>
          <Skeleton h={4} w="40%" borderRadius="md" />
          <SkeletonText noOfLines={3} spacing={2} />
          <Skeleton h={20} borderRadius="md" />
        </VStack>
      )}

      {draftError && (
        <Alert status="warning" borderRadius="md" fontSize="sm">
          <AlertIcon />
          Could not load draft preview: {draftError}
        </Alert>
      )}

      {/* ── Draft content ── */}
      {draft && !draftLoading && (
        <VStack align="stretch" spacing={4}>

          {/* Summary */}
          <Card variant="filled" bg="blue.50" borderColor="blue.200" border="1px solid">
            <CardBody py={3} px={4}>
              <Text fontSize="2xs" fontWeight="bold" textTransform="uppercase" color="blue.400" letterSpacing="wider" mb={1}>
                Summary
              </Text>
              <Text fontSize="sm" color="blue.900" lineHeight="tall">
                {draft.summary}
              </Text>
            </CardBody>
          </Card>

          {/* Editable answer — pre-populated with the generated answer; changes apply on approve */}
          <Box>
            <HStack justify="space-between" mb={2}>
              <Text fontSize="2xs" fontWeight="bold" textTransform="uppercase" color="gray.400" letterSpacing="wider">
                Final Answer
              </Text>
              <Text fontSize="2xs" color="gray.400">
                editable · changes apply on approve · {editedAnswer.length} chars
              </Text>
            </HStack>
            <Textarea
              value={editedAnswer}
              onChange={(e) => setEditedAnswer(e.target.value)}
              size="sm"
              minH="180px"
              resize="vertical"
              focusBorderColor="brand.500"
              bg="white"
              border="1px solid"
              borderColor="gray.200"
              borderRadius="lg"
              fontSize="sm"
              lineHeight="1.8"
              color="gray.700"
              isDisabled={isLoading}
              placeholder="The final answer to deliver…"
            />
          </Box>

          {/* Web search results — always rendered so reviewer knows status */}
          <Box>
            <HStack spacing={2} mb={2} align="center">
              <SearchIcon color="teal.500" boxSize={3} />
              <Text fontSize="2xs" fontWeight="bold" textTransform="uppercase" color="teal.600" letterSpacing="wider">
                Information from the web
              </Text>
              {draft.web_search_results && draft.web_search_results.length > 0 && (
                <Badge colorScheme="teal" variant="subtle" fontSize="2xs" borderRadius="full">
                  {draft.web_search_results.length} results
                </Badge>
              )}
            </HStack>

            {(!draft.web_search_results || draft.web_search_results.length === 0) ? (
              <Box
                bg="gray.50"
                border="1px dashed"
                borderColor="gray.300"
                borderRadius="md"
                px={4}
                py={3}
              >
                <Text fontSize="xs" color="gray.500" fontStyle="italic">
                  No web results available for this query.
                </Text>
              </Box>
            ) : (
              <VStack align="stretch" spacing={2}>
                {draft.web_search_results.map((r, i) => (
                  <Box
                    key={i}
                    bg="teal.50"
                    border="1px solid"
                    borderColor="teal.200"
                    borderRadius="md"
                    p={3}
                  >
                    <HStack justify="space-between" align="start" mb={1} spacing={2}>
                      <Text fontSize="xs" fontWeight="semibold" color="teal.800" noOfLines={1} flex={1}>
                        {r.title || 'Untitled'}
                      </Text>
                      {r.link && (
                        <a href={r.link} target="_blank" rel="noopener noreferrer">
                          <ExternalLinkIcon color="teal.500" boxSize={3} flexShrink={0} />
                        </a>
                      )}
                    </HStack>
                    <Text fontSize="xs" color="teal.700" noOfLines={3} lineHeight="tall">
                      {r.snippet}
                    </Text>
                    {r.link && (
                      <Text fontSize="2xs" color="teal.500" mt={1} noOfLines={1}>
                        {r.link}
                      </Text>
                    )}
                  </Box>
                ))}
              </VStack>
            )}
          </Box>

          {/* Citations */}
          {draft.citations.length > 0 && (
            <Box>
              <Text fontSize="2xs" fontWeight="bold" textTransform="uppercase" color="gray.400" letterSpacing="wider" mb={2}>
                Source Documents ({draft.citations.length})
              </Text>
              <VStack align="stretch" spacing={2}>
                {draft.citations.map((c, i) => (
                  <HStack
                    key={c.document_id}
                    bg="gray.50"
                    border="1px solid"
                    borderColor="gray.200"
                    borderRadius="md"
                    p={3}
                    spacing={3}
                    align="start"
                  >
                    <AttachmentIcon color="gray.400" mt={0.5} flexShrink={0} />
                    <VStack align="start" spacing={0.5} flex={1} minW={0}>
                      <HStack spacing={2} w="full" justify="space-between">
                        <Text fontSize="xs" fontWeight="semibold" color="gray.700" noOfLines={1}>
                          {c.source || c.document_id}
                        </Text>
                        <Badge colorScheme={scoreColor(c.relevance_score)} fontSize="2xs" variant="subtle" flexShrink={0}>
                          #{i + 1} · {Math.round(c.relevance_score * 100)}%
                        </Badge>
                      </HStack>
                      <Text fontSize="xs" color="gray.500" fontStyle="italic" noOfLines={2}>
                        "{c.excerpt}"
                      </Text>
                    </VStack>
                  </HStack>
                ))}
              </VStack>
            </Box>
          )}

          {/* Confidence + groundedness + context precision scores */}
          <ConfidenceStats
            confidence={draft.confidence}
            groundedness={draft.groundedness}
            contextPrecision={draft.context_precision}
          />

        </VStack>
      )}

      <Divider />

      {/* ── Decision form ── */}
      <VStack align="stretch" spacing={4}>
        <FormControl isRequired isInvalid={!!formError && !reviewerId.trim()}>
          <FormLabel fontSize="sm" fontWeight="semibold">Reviewer ID</FormLabel>
          <Input
            value={reviewerId}
            onChange={(e) => setReviewerId(e.target.value)}
            placeholder="e.g. analyst@company.com"
            size="sm"
            focusBorderColor="brand.500"
            bg="white"
            isDisabled={isLoading}
          />
        </FormControl>

        <FormControl>
          <FormLabel fontSize="sm" fontWeight="semibold">
            Comment{' '}
            <Text as="span" fontWeight="normal" color="gray.400">(optional)</Text>
          </FormLabel>
          <Textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Add a note about your decision…"
            size="sm"
            rows={2}
            resize="none"
            focusBorderColor="brand.500"
            bg="white"
            isDisabled={isLoading}
          />
          <FormHelperText>{comment.length} / 1024</FormHelperText>
        </FormControl>
      </VStack>

      {formError && (
        <Alert status="error" borderRadius="md" fontSize="sm">
          <AlertIcon />
          {formError}
        </Alert>
      )}

      <HStack spacing={3} pt={1}>
        <Button
          flex={1}
          colorScheme="green"
          leftIcon={<CheckCircleIcon />}
          onClick={() => handleDecision('approved')}
          isLoading={isLoading && pendingAction === 'approved'}
          loadingText="Approving…"
          isDisabled={isLoading || !reviewerId.trim() || draftLoading}
          size="md"
        >
          Approve
        </Button>
        <Button
          flex={1}
          colorScheme="red"
          variant="outline"
          leftIcon={<CloseIcon boxSize={3} />}
          onClick={() => handleDecision('rejected')}
          isLoading={isLoading && pendingAction === 'rejected'}
          loadingText="Rejecting…"
          isDisabled={isLoading || !reviewerId.trim() || draftLoading}
          size="md"
        >
          Reject
        </Button>
      </HStack>

    </VStack>
  );
}
