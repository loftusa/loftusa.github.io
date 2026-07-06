export function NetworkIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="5.5" cy="6" r="2.2" />
      <circle cx="18.5" cy="6" r="2.2" />
      <circle cx="12" cy="18" r="2.2" />
      <path d="M7.5 7.6 10.8 16M16.5 7.6 13.2 16M7.7 6h8.6" />
    </svg>
  );
}

export function HouseIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3.5 11.5 12 4.5l8.5 7" />
      <path d="M6 10.5V19h12v-8.5" />
    </svg>
  );
}

export function BriefcaseIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="4" y="8" width="16" height="11" rx="1.5" />
      <path d="M9.5 8V6.8A1.8 1.8 0 0 1 11.3 5h1.4a1.8 1.8 0 0 1 1.8 1.8V8" />
    </svg>
  );
}
