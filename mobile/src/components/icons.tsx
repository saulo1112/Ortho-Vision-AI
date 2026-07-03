/** Minimal inline stroke icons (lucide-style) — no icon-font dependency. */
import Svg, { Circle, Path, Polyline, Rect } from 'react-native-svg';

interface IconProps {
  size?: number;
  color?: string;
  strokeWidth?: number;
}

function base({ size = 22, color = '#F1F7FB', strokeWidth = 1.8 }: IconProps) {
  return {
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: color,
    strokeWidth,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
  };
}

export function CameraIcon(props: IconProps) {
  return (
    <Svg {...base(props)}>
      <Path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
      <Circle cx={12} cy={13} r={3} />
    </Svg>
  );
}

export function GalleryIcon(props: IconProps) {
  return (
    <Svg {...base(props)}>
      <Rect x={3} y={3} width={18} height={18} rx={2} />
      <Circle cx={9} cy={9} r={2} />
      <Path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
    </Svg>
  );
}

export function ClockIcon(props: IconProps) {
  return (
    <Svg {...base(props)}>
      <Circle cx={12} cy={12} r={10} />
      <Polyline points="12 6 12 12 16 14" />
    </Svg>
  );
}

export function ChevronLeftIcon(props: IconProps) {
  return (
    <Svg {...base(props)}>
      <Path d="m15 18-6-6 6-6" />
    </Svg>
  );
}

export function PulseIcon(props: IconProps) {
  return (
    <Svg {...base(props)}>
      <Path d="M22 12h-4l-3 9L9 3l-3 9H2" />
    </Svg>
  );
}
