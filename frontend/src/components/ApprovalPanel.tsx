import { useEffect, useState } from 'react';
import {
  Alert,
  AlertIcon,
  Badge,
  Box,
  Button,
  Card,
  CardBody,
  CircularProgress,
  CircularProgressLabel,
  Divider,
  FormControl,
  FormHelperText,
  FormLabel,
  HStack,
  Input,
  Progress,
  Skeleton,
  SkeletonText,
  Text,
  Textarea,
  VStack,
} from '@chakra-ui/react';
import { AttachmentIcon, CheckCircleIcon, CloseIcon } from '@chakra-ui/icons';
import { getWorkflowDraft } from '../api/workflow';
import type { ApprovalAction, DraftResponse } from '../types/workflow';

const ROUTE_LABELS: Record<string, { label: string; color: string }> = {
  research: { label: 'Research Agent', color: 'blue' },
  support:  { label: 'Support Agent',  color: 'teal' },
};

function scoreColor(score: number): string {
  if (score >= 0.75) return 'green';
  if (score >= 0.45) return 'orange';
  return 'red';
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  return (
    <Box>
      <HStack justify="space-between" mb={1}>
        <Text fontSize="2xs" color="gray.500" textTransform="uppercase" letterSpacing="wider">{label}</Text>
        <Text fontSize="2xs" fontWeight="bold" color={`${scoreColor(value)}.600`}>{pct}%</Text>
      </HStack>
      <Progress value={pct} size="xs" colorScheme={scoreColor(value)} borderRadius="full" bg="gray.100" />
    </Box>
  );
}

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

  const route            = draft ? (ROUTE_LABELS[draft.route] ?? { label: draft.route, color: 'gray' }) : null;
  const overallScore     = draft?.confidence?.overall ?? null;
  const groundScore      = draft?.groundedness?.groundedness_score ?? null;
  const unsupported      = draft?.groundedness?.unsupported_claims ?? [];

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

          {/* Confidence + groundedness scores */}
          {(overallScore !== null || groundScore !== null) && (
            <HStack
              spacing={4}
              bg="gray.50"
              border="1px solid"
              borderColor="gray.200"
              borderRadius="lg"
              p={4}
              align="start"
              wrap="wrap"
            >
              {overallScore !== null && (
                <VStack spacing={1} align="center" minW="72px">
                  <CircularProgress
                    value={Math.round(overallScore * 100)}
                    color={`${scoreColor(overallScore)}.400`}
                    trackColor="gray.100"
                    size="56px"
                    thickness="10px"
                  >
                    <CircularProgressLabel fontSize="xs" fontWeight="bold">
                      {Math.round(overallScore * 100)}%
                    </CircularProgressLabel>
                  </CircularProgress>
                  <Text fontSize="2xs" color="gray.500" textAlign="center">Confidence</Text>
                </VStack>
              )}
              {groundScore !== null && (
                <VStack spacing={1} align="center" minW="72px">
                  <CircularProgress
                    value={Math.round(groundScore * 100)}
                    color={`${scoreColor(groundScore)}.400`}
                    trackColor="gray.100"
                    size="56px"
                    thickness="10px"
                  >
                    <CircularProgressLabel fontSize="xs" fontWeight="bold">
                      {Math.round(groundScore * 100)}%
                    </CircularProgressLabel>
                  </CircularProgress>
                  <Text fontSize="2xs" color="gray.500" textAlign="center">Grounded</Text>
                </VStack>
              )}
              {draft.confidence && (
                <VStack align="stretch" spacing={2} flex={1} minW="160px">
                  <ScoreBar label="Routing"   value={draft.confidence.router} />
                  <ScoreBar label="Retrieval" value={draft.confidence.retrieval} />
                  <ScoreBar label="Answer"    value={draft.confidence.answer} />
                </VStack>
              )}
            </HStack>
          )}

          {/* Unsupported claims warning */}
          {unsupported.length > 0 && (
            <Alert status="warning" borderRadius="md" fontSize="sm" alignItems="start">
              <AlertIcon mt={0.5} />
              <Box>
                <Text fontWeight="semibold" mb={1}>
                  {unsupported.length} unsupported claim{unsupported.length > 1 ? 's' : ''} detected
                </Text>
                <VStack align="start" spacing={1}>
                  {unsupported.map((c, i) => (
                    <Text key={i} fontSize="xs" color="orange.800">· {c.claim}</Text>
                  ))}
                </VStack>
              </Box>
            </Alert>
          )}

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
