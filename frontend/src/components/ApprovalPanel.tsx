import { useState } from 'react';
import {
  Alert,
  AlertIcon,
  Box,
  Button,
  Divider,
  FormControl,
  FormHelperText,
  FormLabel,
  HStack,
  Input,
  Text,
  Textarea,
  VStack,
} from '@chakra-ui/react';
import { CheckCircleIcon, CloseIcon } from '@chakra-ui/icons';
import type { ApprovalAction } from '../types/workflow';

interface Props {
  sessionId: string;
  query: string;
  onDecision: (action: ApprovalAction, reviewerId: string, comment?: string) => Promise<void>;
}

export default function ApprovalPanel({ sessionId, query, onDecision }: Props) {
  const [reviewerId, setReviewerId] = useState('');
  const [comment, setComment] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [pendingAction, setPendingAction] = useState<ApprovalAction | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDecision = async (action: ApprovalAction) => {
    if (!reviewerId.trim()) {
      setError('Reviewer ID is required.');
      return;
    }
    setError(null);
    setIsLoading(true);
    setPendingAction(action);
    try {
      await onDecision(action, reviewerId.trim(), comment.trim() || undefined);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Approval failed. Please try again.');
      setIsLoading(false);
      setPendingAction(null);
    }
  };

  return (
    <VStack align="stretch" spacing={5}>
      <Box>
        <Text fontSize="xs" fontWeight="bold" textTransform="uppercase" color="gray.400" letterSpacing="wider" mb={2}>
          Pending Review
        </Text>
        <Text fontSize="sm" color="gray.600" lineHeight="tall">
          The AI agents have processed your query and generated a structured response.
          Review the request below and submit your decision to proceed.
        </Text>
      </Box>

      <Box bg="purple.50" border="1px solid" borderColor="purple.200" borderRadius="lg" p={4}>
        <Text fontSize="2xs" fontWeight="bold" textTransform="uppercase" color="purple.400" letterSpacing="wider" mb={1}>
          Query
        </Text>
        <Text fontSize="sm" color="purple.800" fontStyle="italic">
          "{query}"
        </Text>
        <Text fontSize="2xs" color="gray.400" mt={2}>
          Session: {sessionId}
        </Text>
      </Box>

      <Divider />

      <VStack align="stretch" spacing={4}>
        <FormControl isRequired isInvalid={!!error && !reviewerId.trim()}>
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
            <Text as="span" fontWeight="normal" color="gray.400">
              (optional)
            </Text>
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

      {error && (
        <Alert status="error" borderRadius="md" fontSize="sm">
          <AlertIcon />
          {error}
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
          isDisabled={isLoading || !reviewerId.trim()}
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
          isDisabled={isLoading || !reviewerId.trim()}
          size="md"
        >
          Reject
        </Button>
      </HStack>
    </VStack>
  );
}
