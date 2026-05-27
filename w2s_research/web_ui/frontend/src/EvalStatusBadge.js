import React from 'react';

// Anthropic light theme eval_status colors.
// Distinct from StatusBadge.js (which colours Experiment.status); these
// render the five derived eval_status values on Findings.
const evalStatusConfig = {
  pending: {
    bg: '#DBEAFE',
    color: '#1E40AF',
    label: 'Pending eval',
    icon: 'spinner',
  },
  verified: {
    bg: '#D1FAE5',
    color: '#065F46',
    label: 'Verified',
    icon: null,
  },
  failed: {
    bg: '#FEE2E2',
    color: '#991B1B',
    label: 'Eval failed',
    icon: null,
  },
  not_applicable: {
    bg: '#F3F4F6',
    color: '#6B7280',
    label: '',
    icon: null,
  },
  orphaned: {
    bg: '#FCE7F3',
    color: '#9D174D',
    label: 'Orphaned — report',
    icon: 'wrench',
    tooltip: 'Linked evaluation missing. Report to operator.',
  },
};

const EvalStatusBadge = ({ status, ptScore, hideNotApplicable = true }) => {
  if (!status) return null;
  if (status === 'not_applicable' && hideNotApplicable) return null;

  const config = evalStatusConfig[status] || evalStatusConfig.pending;

  return (
    <span
      title={config.tooltip}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        padding: '4px 10px',
        borderRadius: '6px',
        fontSize: '11px',
        fontWeight: '600',
        textTransform: 'uppercase',
        letterSpacing: '0.04em',
        background: config.bg,
        color: config.color,
      }}
    >
      {config.icon === 'spinner' && (
        <span style={{
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          background: config.color,
          animation: 'eval-pulse 1.5s ease-in-out infinite',
        }} />
      )}
      {config.icon === 'wrench' && (
        <span aria-hidden style={{ fontSize: '12px' }}>⚠</span>
      )}
      {config.label}
      {status === 'verified' && typeof ptScore === 'number' && (
        <span style={{ marginLeft: '4px', fontWeight: '500' }}>
          pt={ptScore.toFixed(3)}
        </span>
      )}
      <style>{`
        @keyframes eval-pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </span>
  );
};

export default EvalStatusBadge;
