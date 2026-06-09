import {
  Badge,
  Box,
  Card,
  CardBody,
  Divider,
  HStack,
  Progress,
  Text,
  VStack,
} from '@chakra-ui/react';
import { AttachmentIcon } from '@chakra-ui/icons';
import type { Citation } from '../types/workflow';

const EXCERPT_MAX = 220;

function truncate(text: string): string {
  return text.length > EXCERPT_MAX ? `${text.slice(0, EXCERPT_MAX)}…` : text;
}

function scoreColor(score: number): string {
  if (score >= 0.75) return 'green';
  if (score >= 0.45) return 'orange';
  return 'red';
}

interface CitationCardProps {
  citation: Citation;
  rank: number;
}

function CitationCard({ citation, rank }: CitationCardProps) {
  const pct = Math.round(citation.relevance_score * 100);
  const color = scoreColor(citation.relevance_score);

  return (
    <Card variant="outline" size="sm" bg="white">
      <CardBody>
        <VStack align="stretch" spacing={2}>
          <HStack justify="space-between" wrap="wrap" gap={1}>
            <HStack spacing={2}>
              <AttachmentIcon color="gray.400" boxSize={3.5} />
              <Text fontSize="xs" fontWeight="semibold" color="gray.700" noOfLines={1}>
                {citation.source || citation.document_id}
              </Text>
            </HStack>
            <Badge colorScheme={color} fontSize="2xs" variant="subtle">
              #{rank} · {pct}% match
            </Badge>
          </HStack>

          <Divider />

          <Text fontSize="xs" color="gray.600" lineHeight="tall" fontStyle="italic">
            "{truncate(citation.excerpt)}"
          </Text>

          <Box>
            <HStack justify="space-between" mb={1}>
              <Text fontSize="2xs" color="gray.400" textTransform="uppercase" letterSpacing="wider">
                Relevance
              </Text>
              <Text fontSize="2xs" color={`${color}.600`} fontWeight="bold">
                {pct}%
              </Text>
            </HStack>
            <Progress
              value={pct}
              size="xs"
              colorScheme={color}
              borderRadius="full"
              bg="gray.100"
            />
          </Box>
        </VStack>
      </CardBody>
    </Card>
  );
}

interface Props {
  citations: Citation[];
}

export default function DocumentsPanel({ citations }: Props) {
  if (citations.length === 0) {
    return (
      <Box p={4} textAlign="center" color="gray.400">
        <Text fontSize="sm">No source documents available.</Text>
      </Box>
    );
  }

  const sorted = [...citations].sort((a, b) => b.relevance_score - a.relevance_score);

  return (
    <VStack align="stretch" spacing={3}>
      <HStack justify="space-between">
        <Text fontSize="xs" fontWeight="bold" textTransform="uppercase" color="gray.400" letterSpacing="wider">
          Reranked Source Documents
        </Text>
        <Badge colorScheme="purple" variant="subtle" fontSize="2xs">
          Top {sorted.length} of 10 retrieved
        </Badge>
      </HStack>
      {sorted.map((c, i) => (
        <CitationCard key={c.document_id} citation={c} rank={i + 1} />
      ))}
    </VStack>
  );
}
