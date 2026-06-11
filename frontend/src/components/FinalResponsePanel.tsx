import {
  Badge,
  Box,
  Card,
  CardBody,
  Divider,
  HStack,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  Text,
  VStack,
} from '@chakra-ui/react';
import { CheckCircleIcon, CheckIcon, InfoIcon } from '@chakra-ui/icons';
import type { WorkflowResponse } from '../types/workflow';
import ConfidenceStats from './ConfidenceStats';
import DocumentsPanel from './DocumentsPanel';
import WorkflowMetricsPanel from './WorkflowMetricsPanel';

const ROUTE_LABELS: Record<string, { label: string; color: string }> = {
  research: { label: 'Research', color: 'blue' },
  support:  { label: 'Support',  color: 'teal' },
};

interface Props {
  result: WorkflowResponse;
}

export default function FinalResponsePanel({ result }: Props) {
  const route = ROUTE_LABELS[result.route] ?? { label: result.route, color: 'gray' };

  return (
    <VStack align="stretch" spacing={5}>

      {/* Approval stats — shown for both auto and manual approval */}
      {(result.confidence || result.groundedness) && (
        <Box>
          <Text
            fontSize="2xs"
            fontWeight="bold"
            textTransform="uppercase"
            letterSpacing="wider"
            mb={3}
            color={result.auto_approved ? 'purple.400' : 'green.500'}
          >
            {result.auto_approved ? 'Auto-approval scores' : 'Review scores'}
          </Text>

          <ConfidenceStats
            confidence={result.confidence}
            groundedness={result.groundedness}
            contextPrecision={result.context_precision}
          />

          {/* Manual approval attribution + KB update notice */}
          {!result.auto_approved && result.reviewer_id && (
            <Box
              mt={3}
              bg="green.50"
              border="1px solid"
              borderColor="green.200"
              borderRadius="lg"
              px={4}
              py={3}
            >
              <VStack align="stretch" spacing={2}>
                <HStack spacing={2} align="start">
                  <CheckIcon color="green.500" mt="3px" flexShrink={0} />
                  <VStack align="start" spacing={0.5}>
                    <Text fontSize="xs" color="green.700">
                      Reviewed and approved by{' '}
                      <Text as="span" fontWeight="bold">{result.reviewer_id}</Text>
                    </Text>
                    {result.reviewer_comment && (
                      <Text fontSize="xs" color="green.600" fontStyle="italic">
                        "{result.reviewer_comment}"
                      </Text>
                    )}
                  </VStack>
                </HStack>

                {result.knowledge_updated && (
                  <HStack
                    spacing={2}
                    align="start"
                    pt={2}
                    borderTop="1px solid"
                    borderColor="green.200"
                  >
                    <InfoIcon color="blue.400" mt="3px" flexShrink={0} />
                    <VStack align="start" spacing={0}>
                      <Text fontSize="xs" fontWeight="semibold" color="blue.700">
                        Answer added to knowledge base
                      </Text>
                      <Text fontSize="xs" color="blue.600">
                        Similar queries may auto-approve next time.
                      </Text>
                    </VStack>
                  </HStack>
                )}
              </VStack>
            </Box>
          )}
        </Box>
      )}

      <HStack justify="space-between" wrap="wrap" gap={2}>
        <HStack spacing={2}>
          <CheckCircleIcon color="green.500" boxSize={5} />
          <Text fontWeight="bold" fontSize="lg" color="gray.800">
            Response Ready
          </Text>
        </HStack>
        <HStack spacing={2}>
          <Badge colorScheme={route.color} variant="subtle" fontSize="xs" px={2} py={0.5} borderRadius="full">
            {route.label} Agent
          </Badge>
          <Badge
            colorScheme={result.auto_approved ? 'purple' : 'green'}
            variant="subtle"
            fontSize="xs"
            px={2}
            py={0.5}
            borderRadius="full"
          >
            {result.auto_approved ? 'Auto-Approved' : 'Approved'}
          </Badge>
        </HStack>
      </HStack>

      <Card variant="filled" bg="brand.50" borderColor="brand.200" border="1px solid">
        <CardBody>
          <Text fontSize="2xs" fontWeight="bold" textTransform="uppercase" color="brand.400" letterSpacing="wider" mb={2}>
            Summary
          </Text>
          <Text fontSize="sm" color="brand.900" lineHeight="tall">
            {result.summary}
          </Text>
        </CardBody>
      </Card>

      <Divider />

      <Tabs variant="soft-rounded" colorScheme="brand" size="sm">
        <TabList mb={4}>
          <Tab>Full Answer</Tab>
          <Tab>
            Source Documents
            {result.citations.length > 0 && (
              <Badge ml={2} colorScheme="purple" variant="subtle" fontSize="2xs" borderRadius="full">
                {result.citations.length}
              </Badge>
            )}
          </Tab>
        </TabList>

        <TabPanels>
          <TabPanel px={0} pb={0}>
            <Box
              bg="white"
              border="1px solid"
              borderColor="gray.200"
              borderRadius="lg"
              p={5}
            >
              <Text fontSize="sm" color="gray.700" lineHeight="1.8" whiteSpace="pre-wrap">
                {result.answer}
              </Text>
            </Box>

            {result.citations.length > 0 && (
              <Box mt={4}>
                <Text fontSize="xs" color="gray.400" fontStyle="italic">
                  Based on {result.citations.length} source document{result.citations.length !== 1 ? 's' : ''}.
                  See the &quot;Source Documents&quot; tab for details.
                </Text>
              </Box>
            )}
          </TabPanel>

          <TabPanel px={0} pb={0}>
            <DocumentsPanel citations={result.citations} />
          </TabPanel>
        </TabPanels>
      </Tabs>

      {/* Observability metrics — isolated section below the response */}
      {result.metrics && (
        <>
          <Divider />
          <WorkflowMetricsPanel metrics={result.metrics} />
        </>
      )}
    </VStack>
  );
}
