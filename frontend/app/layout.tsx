import "./globals.css";

export const metadata = {
  title: "Soil Analytics",
  description: "Farmer-friendly soil health analysis",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
