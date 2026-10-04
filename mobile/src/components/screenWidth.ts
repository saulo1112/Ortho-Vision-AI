import { createContext, useContext } from 'react';
import { useWindowDimensions } from 'react-native';

/** Set by PhoneFrame.web so layouts size to the mockup, not the browser window. */
export const ScreenWidthContext = createContext<number | null>(null);

export function useScreenWidth(): number {
  const framed = useContext(ScreenWidthContext);
  const { width } = useWindowDimensions();
  return framed ?? width;
}
