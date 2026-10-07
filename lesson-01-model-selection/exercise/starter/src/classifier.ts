import Anthropic from '@anthropic-ai/sdk';
import { MODELS } from './models';
import { tickets } from './sample-tickets';
import { calculateCost, displayComparison, logStats } from './helpers';

// Initialize the Anthropic client
const anthropic = new Anthropic();

/**
 * Helper function to call the Claude API and track execution metrics.
 */
async function callClaude(
  model: string,
  systemPrompt: string,
  userContent: string
) {
  const startTime = Date.now();

  const response = await anthropic.messages.create({
    model: model,
    max_tokens: 1024,
    system: systemPrompt,
    messages: [{ role: 'user', content: userContent }],
  });

  const durationMs = Date.now() - startTime;
  const inputTokens = response.usage.input_tokens;
  const outputTokens = response.usage.output_tokens;
  const cost = calculateCost(model, inputTokens, outputTokens);

  const textOutput =
    response.content[0].type === 'text' ? response.content[0].text : '';

  return {
    model,
    output: textOutput,
    stats: {
      durationMs,
      inputTokens,
      outputTokens,
      cost,
    },
  };
}

// ============================================================================
// STEP 1: Fast & Low Cost Classification using Haiku
// ============================================================================
export async function testHaiku() {
  console.log('\n--- Testing Haiku (Simple Priority Classification) ---');

  const systemPrompt =
    'You are an automated support ticket priority classifier. ' +
    'Classify the ticket into exactly one of these priorities: LOW, MEDIUM, HIGH, URGENT. ' +
    'Respond ONLY with the priority word and nothing else.';

  const result = await callClaude(
    MODELS.HAIKU,
    systemPrompt,
    tickets.simple.content
  );

  console.log(`Ticket Input: "${tickets.simple.content}"`);
  console.log(`Output: ${result.output.trim()}`);
  logStats(result.stats);

  return result;
}

// ============================================================================
// STEP 2: Balanced Analysis using Sonnet
// ============================================================================
export async function testSonnet() {
  console.log('\n--- Testing Sonnet (Structured Issue Analysis) ---');

  const systemPrompt =
    'Analyze the support ticket and extract the following information:\n' +
    '1. Priority Level (LOW, MEDIUM, HIGH, URGENT)\n' +
    '2. Issue Category\n' +
    '3. Key Details\n' +
    '4. Recommended Action\n\n' +
    'Provide a structured summary.';

  const result = await callClaude(
    MODELS.SONNET,
    systemPrompt,
    tickets.moderate.content
  );

  console.log(`Ticket Input: "${tickets.moderate.content}"`);
  console.log(`Output:\n${result.output}`);
  logStats(result.stats);

  return result;
}

// ============================================================================
// STEP 3: Deep Reasoning & Action Plan using Opus
// ============================================================================
export async function testOpus() {
  console.log('\n--- Testing Opus (Complex Strategic Resolution) ---');

  const systemPrompt =
    'Act as a Senior Support Manager. Provide a comprehensive analysis of the ticket including:\n' +
    '1. Issue Summary\n' +
    '2. Root Cause Hypothesis\n' +
    '3. Business & Operational Impact Assessment\n' +
    '4. Prioritized Action Plan (Immediate, Short-term, Long-term)';

  const result = await callClaude(
    MODELS.OPUS,
    systemPrompt,
    tickets.complex.content
  );

  console.log(`Ticket Input: "${tickets.complex.content}"`);
  console.log(`Output:\n${result.output}`);
  logStats(result.stats);

  return result;
}

// ============================================================================
// STEP 4: Side-by-Side Comparison
// ============================================================================
export async function testCompare() {
  console.log('\n--- Comparing All Models on Moderate Ticket ---');

  const systemPrompt =
    'Analyze this ticket and provide: Priority, Main Issue, and 1 Recommended Action.';

  const modelsToTest = [MODELS.HAIKU, MODELS.SONNET, MODELS.OPUS];
  const results = [];

  for (const model of modelsToTest) {
    const res = await callClaude(model, systemPrompt, tickets.moderate.content);
    results.push(res);
  }

  displayComparison(results);
}

// Main runner - uncomment steps as you complete/test them
async function main() {
  try {
    await testHaiku();
    await testSonnet();
    await testOpus();
    await testCompare();
  } catch (error) {
    console.error('Error running model tests:', error);
  }
}

// Execute script if run directly
if (require.main === module) {
  main();
}
