import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Alert,
  AlertIcon,
  Box,
  Card,
  CardBody,
  Container,
  Divider,
  Flex,
  Grid,
  HStack,
  Text,
  VStack,
} from '@chakra-ui/react';
import BotIcon from '../components/BotIcon';
import QueryForm from '../components/QueryForm';
import { submitWorkflow } from '../api/workflow';

const FEATURES = [
  { icon: '🔍', title: 'Multi-Agent Routing', desc: 'Intelligent routing to research or support agents' },
  { icon: '📄', title: 'RAG Pipeline', desc: 'Retrieval-augmented generation with reranking' },
  { icon: '✅', title: 'Human-in-the-Loop', desc: 'Mandatory approval gate before final response' },
  { icon: '🔬', title: 'Structured Output', desc: 'Pydantic-validated responses with citations' },
];

export default function HomePage() {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleSubmit = async (query: string) => {
    setIsLoading(true);
    setSubmitError(null);
    try {
      const { session_id } = await submitWorkflow(query);
      navigate(`/workflow/${session_id}`, { state: { query } });
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Failed to start workflow.');
      setIsLoading(false);
    }
  };

  return (
    <Flex direction="column" minH="100vh">
      {/* Header */}
      <Box bg="brand.600" py={4} px={6} shadow="md">
        <HStack spacing={3}>
          <BotIcon color="white" boxSize={5} />
          <Text fontWeight="bold" fontSize="lg" color="white" letterSpacing="tight">
            Agentic Workflow
          </Text>
          <Text fontSize="sm" color="brand.200">
            — LangGraph · RAG · Human-in-the-Loop
          </Text>
        </HStack>
      </Box>

      {/* Hero */}
      <Box bg="brand.700" pt={16} pb={20} px={6} textAlign="center">
        <Text
          fontSize={{ base: '3xl', md: '4xl' }}
          fontWeight="extrabold"
          color="white"
          lineHeight="shorter"
          mb={3}
        >
          Ask anything.
        </Text>
        <Text fontSize={{ base: 'lg', md: 'xl' }} color="brand.200" maxW="540px" mx="auto">
          A production-style multi-agent AI workflow with retrieval, reranking, and
          human approval built on LangGraph.
        </Text>
      </Box>

      {/* Query form card */}
      <Container maxW="2xl" px={{ base: 4, md: 6 }} mt={-8} mb={12} position="relative" zIndex={1}>
        <Card shadow="xl" borderRadius="2xl" overflow="hidden">
          <CardBody p={8}>
            {submitError && (
              <Alert status="error" borderRadius="md" mb={5} fontSize="sm">
                <AlertIcon />
                {submitError}
              </Alert>
            )}
            <QueryForm onSubmit={handleSubmit} isLoading={isLoading} />
          </CardBody>
        </Card>
      </Container>

      {/* Feature grid */}
      <Container maxW="2xl" px={{ base: 4, md: 6 }} pb={16}>
        <Divider mb={10} />
        <Grid templateColumns={{ base: '1fr 1fr', md: 'repeat(4, 1fr)' }} gap={6}>
          {FEATURES.map((f) => (
            <VStack key={f.title} spacing={2} textAlign="center" p={3}>
              <Text fontSize="2xl">{f.icon}</Text>
              <Text fontWeight="semibold" fontSize="sm" color="gray.700">
                {f.title}
              </Text>
              <Text fontSize="xs" color="gray.500" lineHeight="tall">
                {f.desc}
              </Text>
            </VStack>
          ))}
        </Grid>
      </Container>

      {/* Footer */}
      <Box mt="auto" py={4} borderTop="1px solid" borderColor="gray.200" textAlign="center">
        <Text fontSize="xs" color="gray.400">
          Powered by LangGraph · Ollama · ChromaDB · FastAPI
        </Text>
      </Box>
    </Flex>
  );
}
