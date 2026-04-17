/**
 * Friendly display names for known model identifiers.
 */
const MODEL_NAMES = {
  'llama-3.1-8b-instant': 'Llama 3.1 8B',
  'llama-3.3-70b-versatile': 'Llama 3.3 70B',
  'openai/gpt-oss-20b': 'GPT OSS 20B',
  'openai/gpt-oss-120b': 'GPT OSS 120B',
  'chairman': 'Chairman',
};

/**
 * Returns a human-readable name for a model identifier.
 * Falls back to the part after the last '/' or the full ID.
 */
export function getModelName(model) {
  if (!model) return 'Unknown';
  return MODEL_NAMES[model] ?? (model.split('/').pop() || model);
}
