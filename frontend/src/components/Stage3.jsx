import ReactMarkdown from 'react-markdown';
import { getModelName } from '../utils';
import './Stage3.css';

export default function Stage3({ finalResponse, streaming }) {
  if (!finalResponse && !streaming) return null;

  const content = finalResponse?.response || '';
  const modelName = getModelName(finalResponse?.model || 'chairman');

  return (
    <div className="stage stage3">
      <h3 className="stage-title">Stage 3: Final Council Answer</h3>
      <div className="final-response">
        <div className="chairman-label">
          Chairman: {modelName}
          {streaming && <span className="streaming-indicator"> ●</span>}
        </div>
        <div className="final-text markdown-content">
          <ReactMarkdown>{content}</ReactMarkdown>
          {streaming && <span className="cursor-blink">▋</span>}
        </div>
      </div>
    </div>
  );
}
