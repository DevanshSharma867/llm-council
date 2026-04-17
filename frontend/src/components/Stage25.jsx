import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { getModelName } from '../utils';
import './Stage25.css';

export default function Stage25({ debate, loading }) {
  const [view, setView] = useState('critiques'); // 'critiques' | 'defense'

  if (loading) {
    return (
      <div className="stage stage25">
        <h3 className="stage-title">Stage 2.5: Debate</h3>
        <div className="debate-loading">
          <div className="spinner"></div>
          <span>Running debate — peers are critiquing the top response...</span>
        </div>
      </div>
    );
  }

  if (!debate) return null;

  const { top_model, critiques = [], defense } = debate;

  return (
    <div className="stage stage25">
      <h3 className="stage-title">Stage 2.5: Debate</h3>

      <div className="debate-header">
        <span className="debate-champion-label">Top-ranked model:</span>
        <span className="debate-champion">{getModelName(top_model)}</span>
      </div>

      <p className="stage-description">
        Peers critique the top-ranked response, then that model defends its answer.
        The Chairman uses this debate when forming the final synthesis.
      </p>

      <div className="debate-tabs">
        <button
          className={`debate-tab ${view === 'critiques' ? 'active' : ''}`}
          onClick={() => setView('critiques')}
        >
          Critiques ({critiques.length})
        </button>
        <button
          className={`debate-tab ${view === 'defense' ? 'active' : ''}`}
          onClick={() => setView('defense')}
          disabled={!defense?.content}
        >
          Defense
        </button>
      </div>

      {view === 'critiques' && (
        <div className="debate-critiques">
          {critiques.length === 0 ? (
            <p className="no-critiques">No critiques were returned.</p>
          ) : (
            critiques.map((c, i) => (
              <div key={i} className="critique-card">
                <div className="critique-author">{getModelName(c.model)}</div>
                <div className="critique-text markdown-content">
                  <ReactMarkdown>{c.critique}</ReactMarkdown>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {view === 'defense' && defense?.content && (
        <div className="defense-card">
          <div className="defense-author">{getModelName(defense.model)} defends:</div>
          <div className="defense-text markdown-content">
            <ReactMarkdown>{defense.content}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}
