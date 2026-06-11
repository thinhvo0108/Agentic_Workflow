import {
  Alert,
  AlertIcon,
  Box,
  CircularProgress,
  CircularProgressLabel,
  HStack,
  Progress,
  Text,
  VStack,
} from '@chakra-ui/react';
import type { ConfidenceScores, ContextPrecisionResult, GroundednessResult } from '../types/workflow';

export function scoreColor(score: number): string {
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
  confidence: ConfidenceScores | null;
  groundedness: GroundednessResult | null;
  contextPrecision?: ContextPrecisionResult | null;
}

export default function ConfidenceStats({ confidence, groundedness, contextPrecision }: Props) {
  const overallScore   = confidence?.overall ?? null;
  const groundScore    = groundedness?.groundedness_score ?? null;
  const precisionScore = contextPrecision?.context_precision_score ?? null;
  const unsupported    = groundedness?.unsupported_claims ?? [];
  const irrelevant     = contextPrecision?.irrelevant_documents ?? [];

  if (overallScore === null && groundScore === null && precisionScore === null) return null;

  return (
    <VStack align="stretch" spacing={3}>
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

        {precisionScore !== null && (
          <VStack spacing={1} align="center" minW="72px">
            <CircularProgress
              value={Math.round(precisionScore * 100)}
              color={`${scoreColor(precisionScore)}.400`}
              trackColor="gray.100"
              size="56px"
              thickness="10px"
            >
              <CircularProgressLabel fontSize="xs" fontWeight="bold">
                {Math.round(precisionScore * 100)}%
              </CircularProgressLabel>
            </CircularProgress>
            <Text fontSize="2xs" color="gray.500" textAlign="center">Ctx Precision</Text>
          </VStack>
        )}

        {confidence && (
          <VStack align="stretch" spacing={2} flex={1} minW="160px">
            <ScoreBar label="Routing"   value={confidence.router} />
            <ScoreBar label="Retrieval" value={confidence.retrieval} />
            <ScoreBar label="Answer"    value={confidence.answer} />
          </VStack>
        )}
      </HStack>

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

      {irrelevant.length > 0 && (
        <Alert status="info" borderRadius="md" fontSize="sm" alignItems="start">
          <AlertIcon mt={0.5} />
          <Box>
            <Text fontWeight="semibold" mb={1}>
              {irrelevant.length} irrelevant doc{irrelevant.length > 1 ? 's' : ''} retrieved
            </Text>
            <VStack align="start" spacing={1}>
              {irrelevant.map((v, i) => (
                <Text key={i} fontSize="xs" color="blue.800">· {v.document_id}: {v.reasoning}</Text>
              ))}
            </VStack>
          </Box>
        </Alert>
      )}
    </VStack>
  );
}
