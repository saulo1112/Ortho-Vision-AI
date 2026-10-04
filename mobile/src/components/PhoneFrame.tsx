import type { ReactNode } from 'react';

/** Native: nothing to frame. The desktop mockup lives in PhoneFrame.web.tsx. */
export function PhoneFrame({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
