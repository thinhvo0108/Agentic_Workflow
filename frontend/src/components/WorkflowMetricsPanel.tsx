import {
  Badge,
  Box,
  Divider,
  Grid,
  GridItem,
  HStack,
  Progress,
  Text,
  Tooltip,
  VStack,
} from '@chakra-ui/react';
import { InfoIcon, TimeIcon, WarningIcon } from '@chakra-ui/icons';
import type { WorkflowMetrics } from '../types/workflow';

interface MetricCardProps {
  label: string;
  value: string;
  sub?: string;
  status?: 'good' | 'warn' | 'bad' | 'neutral';
  tooltip?: string;
}

function MetricCard({ label, value, sub, status = 'neutral', tooltip }: MetricCardProps) {
  const borderColor = {
    good: 'green.200',
    warn: 'orange.200',
    bad: 'red.200',
    neutral: 'gray.200',
  }[status];

  const valueColor = {
    good: 'green.600',
    warn: 'orange.500',
    bad: 'red.500',
    neutral: 'gray.700',
  }[status];

  return (
    <Box
      border="1px solid"
      borderColor={borderColor}
      borderRadius="lg"
      p={4}
      bg="white"
      minW={0}
    >
      <VStack align="start" spacing={1}>
        <HStack spacing={1}>
          <Text fontSize="2xs" fontWeight="bold" textTransform="uppercase" letterSpacing="wider" color="gray.400">
            {label}
          </Text>
          {tooltip && (
            <Tooltip label={tooltip} fontSize="xs" hasArrow placement="top">
              <InfoIcon boxSize="10px" color="gray.300" cursor="help" />
            </Tooltip>
          )}
        </HStack>
        <Text fontSize="xl" fontWeight="bold" color={valueColor} lineHeight="1">
          {value}
        </Text>
        {sub && (
          <Text fontSize="xs" color="gray.400">
            {sub}
          </Text>
        )}
      </VStack>
    </Box>
  );
}

interface RateBarProps {
  label: string;
  rate: number | null;
  tooltip: string;
  invert?: boolean;
}

function RateBar({ label, rate, tooltip, invert = false }: RateBarProps) {
  if (rate === null) {
    return (
      <HStack justify="space-between">
        <Text fontSize="xs" color="gray.500">{label}</Text>
        <Badge colorScheme="gray" variant="subtle" fontSize="2xs">N/A</Badge>
      </HStack>
    );
  }

  const pct = Math.round(rate * 100);
  const colorScheme = invert
    ? (pct === 0 ? 'green' : pct <= 10 ? 'yellow' : 'red')
    : (pct === 0 ? 'green' : pct <= 10 ? 'yellow' : 'red');

  return (
    <Tooltip label={tooltip} fontSize="xs" hasArrow placement="top">
      <VStack align="stretch" spacing={1} cursor="help">
        <HStack justify="space-between">
          <Text fontSize="xs" color="gray.600">{label}</Text>
          <Badge
            colorScheme={colorScheme}
            variant="subtle"
            fontSize="2xs"
          >
            {pct}%
          </Badge>
        </HStack>
        <Progress
          value={pct}
          size="xs"
          colorScheme={colorScheme}
          borderRadius="full"
          bg="gray.100"
        />
      </VStack>
    </Tooltip>
  );
}

interface Props {
  metrics: WorkflowMetrics;
}

export default function WorkflowMetricsPanel({ metrics }: Props) {
  const latencyS = (metrics.latency_ms / 1000).toFixed(1);
  const tokenDisplay = metrics.total_tokens > 0
    ? metrics.total_tokens.toLocaleString()
    : '—';
  const tokenSub = metrics.total_tokens > 0 ? 'tokens' : 'not reported by model';

  const latencyStatus: 'good' | 'warn' | 'bad' =
    metrics.latency_ms < 10_000 ? 'good' : metrics.latency_ms < 30_000 ? 'warn' : 'bad';

  const errorStatus: 'good' | 'warn' | 'bad' =
    metrics.error_count === 0 ? 'good' : metrics.error_count <= 1 ? 'warn' : 'bad';

  const judgeStatus: 'good' | 'warn' | 'bad' | 'neutral' =
    metrics.judge_score === null ? 'neutral'
    : metrics.judge_score >= 0.70 ? 'good'
    : metrics.judge_score >= 0.50 ? 'warn'
    : 'bad';

  const judgeDisplay = metrics.judge_score !== null
    ? `${Math.round(metrics.judge_score * 100)}%`
    : '—';

  return (
    <Box>
      <HStack spacing={2} mb={3}>
        <TimeIcon color="gray.400" boxSize="14px" />
        <Text
          fontSize="2xs"
          fontWeight="bold"
          textTransform="uppercase"
          letterSpacing="wider"
          color="gray.400"
        >
          Workflow Metrics
        </Text>
        <Badge colorScheme="gray" variant="outline" fontSize="2xs">
          {metrics.step_count} nodes
        </Badge>
      </HStack>

      {/* Top row — four headline cards */}
      <Grid templateColumns="repeat(4, 1fr)" gap={3} mb={4}>
        <GridItem>
          <MetricCard
            label="Latency"
            value={`${latencyS}s`}
            sub={`${Math.round(metrics.latency_ms)} ms`}
            status={latencyStatus}
            tooltip="Wall-clock time from query submission to final response"
          />
        </GridItem>
        <GridItem>
          <MetricCard
            label="Tokens Used"
            value={tokenDisplay}
            sub={tokenSub}
            status="neutral"
            tooltip="Sum of prompt + completion tokens across all LLM calls (router, generator, groundedness, judge)"
          />
        </GridItem>
        <GridItem>
          <MetricCard
            label="Errors"
            value={metrics.error_count === 0 ? 'None' : String(metrics.error_count)}
            sub={`${Math.round(metrics.error_rate * 100)}% of nodes`}
            status={errorStatus}
            tooltip="Number of nodes that encountered and recovered from errors"
          />
        </GridItem>
        <GridItem>
          <MetricCard
            label="Judge Score"
            value={judgeDisplay}
            sub={metrics.judge_score !== null ? metrics.judge_score >= 0.70 ? 'passed threshold' : 'below threshold' : 'not evaluated'}
            status={judgeStatus}
            tooltip="LLM-as-a-judge overall quality score (faithfulness 40%, relevance 30%, completeness 20%, coherence 10%)"
          />
        </GridItem>
      </Grid>

      {/* Quality rates bar section */}
      <Box
        border="1px solid"
        borderColor="gray.100"
        borderRadius="lg"
        p={4}
        bg="gray.50"
      >
        <HStack spacing={1} mb={3}>
          <WarningIcon boxSize="11px" color="gray.400" />
          <Text fontSize="2xs" fontWeight="bold" textTransform="uppercase" letterSpacing="wider" color="gray.400">
            Quality Signals
          </Text>
        </HStack>
        <VStack align="stretch" spacing={3}>
          <RateBar
            label="Hallucination Rate"
            rate={metrics.hallucination_rate}
            tooltip={
              metrics.hallucination_rate !== null
                ? `${Math.round((metrics.hallucination_rate) * 100)}% of factual claims in the answer were not supported by source documents (lower is better)`
                : 'Groundedness evaluation did not run for this request'
            }
          />
          <Divider />
          <RateBar
            label="Error Rate"
            rate={metrics.error_rate}
            tooltip={`${Math.round(metrics.error_rate * 100)}% of workflow nodes encountered errors (all were recovered)`}
          />
        </VStack>
      </Box>
    </Box>
  );
}
