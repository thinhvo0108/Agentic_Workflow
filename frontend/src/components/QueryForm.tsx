import { useState } from 'react';
import {
  Box,
  Button,
  FormControl,
  FormErrorMessage,
  FormHelperText,
  FormLabel,
  Textarea,
  VStack,
} from '@chakra-ui/react';
import { ArrowForwardIcon } from '@chakra-ui/icons';

interface Props {
  onSubmit: (query: string) => Promise<void>;
  isLoading: boolean;
}

const MAX_LENGTH = 4096;
const UNSAFE_RE = /[<>{};`$]/;

export default function QueryForm({ onSubmit, isLoading }: Props) {
  const [query, setQuery] = useState('');
  const [error, setError] = useState<string | null>(null);

  const validate = (value: string): string | null => {
    if (!value.trim()) return 'Query cannot be empty.';
    if (value.length > MAX_LENGTH) return `Query must be under ${MAX_LENGTH} characters.`;
    if (UNSAFE_RE.test(value)) return 'Query contains disallowed characters.';
    return null;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const err = validate(query);
    if (err) { setError(err); return; }
    setError(null);
    await onSubmit(query.trim());
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setQuery(e.target.value);
    if (error) setError(validate(e.target.value));
  };

  return (
    <Box as="form" onSubmit={handleSubmit} w="full">
      <VStack spacing={4} align="stretch">
        <FormControl isInvalid={!!error} isRequired>
          <FormLabel fontWeight="semibold" color="gray.700">
            Your question
          </FormLabel>
          <Textarea
            value={query}
            onChange={handleChange}
            placeholder="e.g. What are the key technical differences between transformer and LSTM architectures?"
            rows={4}
            resize="vertical"
            focusBorderColor="brand.500"
            bg="white"
            fontSize="md"
            isDisabled={isLoading}
          />
          {error ? (
            <FormErrorMessage>{error}</FormErrorMessage>
          ) : (
            <FormHelperText color="gray.500">
              {query.length} / {MAX_LENGTH} characters
            </FormHelperText>
          )}
        </FormControl>

        <Button
          type="submit"
          size="lg"
          colorScheme="brand"
          rightIcon={<ArrowForwardIcon />}
          isLoading={isLoading}
          loadingText="Submitting…"
          isDisabled={!query.trim() || isLoading}
          alignSelf="flex-end"
          px={8}
        >
          Run workflow
        </Button>
      </VStack>
    </Box>
  );
}
