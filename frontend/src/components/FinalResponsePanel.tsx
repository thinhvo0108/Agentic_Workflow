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
import { CheckCircleIcon } from '@chakra-ui/icons';
import type { WorkflowResponse } from '../types/workflow';
import DocumentsPanel from './DocumentsPanel';

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
          <Badge colorScheme="green" variant="subtle" fontSize="xs" px={2} py={0.5} borderRadius="full">
            Approved
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
    </VStack>
  );
}
